from contextlib import asynccontextmanager
import os
from typing import Any
from typing import Optional, List

from sqlalchemy.sql.expression import BinaryExpression
from sqlalchemy import create_engine, delete, func, select, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, joinedload
from enum import Enum

from common_tools.helpers.txt_helper import txt
from common_tools.helpers.file_helper import file

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

class GenericDataContext:
    def __init__(self, base_entities, db_path_or_url='database.db', log_queries_to_terminal=False, db_type: DatabaseType | None = None):
        """
        Initialize GenericDataContext with support for SQLite and PostgreSQL.

        Args:
            base_entities: SQLAlchemy declarative base with entity mappings
            db_path_or_url: Database path (SQLite) or connection URL (PostgreSQL)
                - SQLite: 'database.db' or 'path/to/database.db'
                - PostgreSQL: 'postgresql+asyncpg://user:password@host:port/dbname'
            log_queries_to_terminal: Enable SQLAlchemy query logging
            db_type: DatabaseType.SQLITE or DatabaseType.POSTGRESQL (auto-detected from URL)
        """
        self.base_entities = base_entities
        self.log_queries_to_terminal = log_queries_to_terminal

        # Auto-detect database type from URL if not explicitly specified
        if not db_type:
            if 'postgresql' in db_path_or_url:
                db_type = DatabaseType.POSTGRESQL
            else:
                db_type = DatabaseType.SQLITE
        self.db_type = db_type

        # Configure database URLs based on type
        if self.db_type == DatabaseType.POSTGRESQL:
            self._configure_postgresql_async(db_path_or_url)
        else:
            self._configure_sqlite_async(db_path_or_url)

        # Create async engine
        self.engine = create_async_engine(self.async_db_url, echo=log_queries_to_terminal)
        self.local_session = sessionmaker(
                                bind=self.engine,
                                expire_on_commit=False,
                                class_=AsyncSession)

    def _configure_sqlite_async(self, db_path_or_url: str):
        """Configure SQLite database paths."""
        db_full_path_and_name = None
        if 'http' not in db_path_or_url and not db_path_or_url.startswith('sqlite'):
            source_path = os.environ.get("PYTHONPATH", "").split(';')[-1]
            db_full_path_and_name = os.path.join(source_path.replace('\\', '/'), db_path_or_url.replace('\\', '/')).replace('\\', '/')

        self.db_path_or_url = db_full_path_and_name or db_path_or_url

        # Remove sqlite:/// or sqlite+aiosqlite:/// prefix if present for file path checking
        clean_path = self.db_path_or_url
        if clean_path.startswith('sqlite+aiosqlite:///'):
            clean_path = clean_path.replace('sqlite+aiosqlite:///', '')
        elif clean_path.startswith('sqlite:///'):
            clean_path = clean_path.replace('sqlite:///', '')

        # Check if database file exists (only for file-based SQLite, skip for :memory:)
        if 'http' not in clean_path and clean_path != ':memory:' and not file.exists(clean_path):
            txt.print(f"/!\\ Database file not found at path: {clean_path}")
            self.create_database()

        self.sync_db_url = f'sqlite:///{clean_path}' if not clean_path.startswith('sqlite') else self.db_path_or_url
        self.async_db_url = self.sync_db_url.replace('sqlite:///', 'sqlite+aiosqlite:///')

    def _configure_postgresql_async(self, db_url: str):
        """Configure PostgreSQL database URLs."""
        self.db_path_or_url = db_url

        # Ensure async driver (asyncpg)
        if 'postgresql+asyncpg://' not in db_url:
            if db_url.startswith('postgresql://'):
                self.async_db_url = db_url.replace('postgresql://', 'postgresql+asyncpg://')
            else:
                self.async_db_url = f'postgresql+asyncpg://{db_url}'
        else:
            self.async_db_url = db_url

        # Sync URL for database creation (using psycopg2)
        self.sync_db_url = self.async_db_url.replace('postgresql+asyncpg://', 'postgresql+psycopg2://')

        # Check if PostgreSQL database tables exist, create if not
        self._create_missing_tables_postgre()

    def _create_missing_tables_postgre(self):
        """Check if tables exist in PostgreSQL database and create if needed."""
        try:
            sync_engine = create_engine(self.sync_db_url, echo=False)
            with sync_engine.connect():
                # Check if any tables exist from our metadata using inspect
                inspector = inspect(sync_engine)
                existing_tables = inspector.get_table_names()

                # Get expected tables from metadata
                expected_tables = [table.name for table in self.base_entities.metadata.sorted_tables]

                # If no expected tables exist, create all tables
                if not any(table in existing_tables for table in expected_tables):
                    txt.print("/!\\ PostgreSQL tables not found, creating schema...")
                    self.create_database()
                else:
                    # Check for missing tables and create them
                    missing_tables = [table for table in expected_tables if table not in existing_tables]
                    if missing_tables:
                        txt.print(f"/!\\ Missing tables detected: {missing_tables}, creating them...")
                        self.create_missing_tables_only()
        except Exception as e:
            txt.print(f"/!\\ Could not verify PostgreSQL tables: {e}")
            txt.print("Attempting to create tables...")
            try:
                self.create_database()
            except Exception as create_error:
                txt.print(f"/!\\ Failed to create PostgreSQL tables: {create_error}")
                raise

    def drop_all_tables_postgre(self):
        """Drop all tables from PostgreSQL database. PostgreSQL only.

        WARNING: This will permanently delete all tables and data.
        Use with extreme caution, primarily for development/testing.

        Raises:
            ValueError: If called on non-PostgreSQL database
        """
        if self.db_type != DatabaseType.POSTGRESQL:
            raise ValueError("drop_all_tables() is only supported for PostgreSQL databases")

        txt.print(">>> WARNING: Dropping all tables from PostgreSQL database")
        try:
            sync_engine = create_engine(self.sync_db_url, echo=True)
            with sync_engine.begin() as conn:
                # Drop all tables in reverse order to handle foreign key constraints
                self.base_entities.metadata.drop_all(bind=conn, checkfirst=True)
            txt.print(">>> All tables dropped successfully")
        except Exception as e:
            txt.print(f"/!\\ Failed to drop PostgreSQL tables: {e}")
            raise

    def create_database(self):
        """Create database and tables (synchronously). Handles both SQLite and PostgreSQL."""
        txt.print(">>> Creating database & tables")

        if self.db_type == DatabaseType.SQLITE:
            # Only create directory for SQLite file-based databases
            if 'http' not in self.db_path_or_url:
                parent_dir = os.path.dirname(self.db_path_or_url)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    txt.print(f"Created directory: {parent_dir}")

            # Create sync engine and tables for SQLite
            sync_engine = create_engine(self.sync_db_url, echo=True)
            with sync_engine.begin() as conn:
                self.base_entities.metadata.create_all(bind=conn)
            txt.print(">>> Database & tables creation completed successfully.")

        elif self.db_type == DatabaseType.POSTGRESQL:
            # For PostgreSQL, assume the database already exists on the server
            # Only create tables/schema
            txt.print(">>> Creating tables in existing PostgreSQL database")
            try:
                sync_engine = create_engine(self.sync_db_url, echo=True)
                with sync_engine.begin() as conn:
                    self.base_entities.metadata.create_all(bind=conn)
                txt.print(">>> PostgreSQL tables creation completed successfully.")
            except Exception as e:
                txt.print(f"/!\\ Failed to create PostgreSQL tables: {e}")
                txt.print("Ensure the database exists and credentials are correct.")
                raise

    def create_missing_tables_only(self):
        """Create only missing tables without dropping existing ones. PostgreSQL only.

        This method uses create_all with checkfirst=True to safely add new tables
        to an existing database schema without affecting existing tables.
        """
        if self.db_type != DatabaseType.POSTGRESQL:
            txt.print("/!\\ create_missing_tables_only() is primarily for PostgreSQL, but will attempt on current database type")

        txt.print(">>> Creating missing tables only (preserving existing data)")
        try:
            sync_engine = create_engine(self.sync_db_url, echo=True)
            with sync_engine.begin() as conn:
                self.base_entities.metadata.create_all(bind=conn, checkfirst=True)
            txt.print(">>> Missing tables created successfully")
        except Exception as e:
            txt.print(f"/!\\ Failed to create missing tables: {e}")
            raise

    async def create_database_async(self):
        """Create database and tables asynchronously. Handles both SQLite and PostgreSQL.

        Use this method when working with async SQLite databases to ensure tables
        are visible to async sessions immediately.
        """
        txt.print(">>> Creating database & tables (async)")

        if self.db_type == DatabaseType.SQLITE:
            # Only create directory for SQLite file-based databases
            if 'http' not in self.db_path_or_url and self.db_path_or_url != ':memory:':
                parent_dir = os.path.dirname(self.db_path_or_url)
                if parent_dir and not os.path.exists(parent_dir):
                    os.makedirs(parent_dir, exist_ok=True)
                    txt.print(f"Created directory: {parent_dir}")

        # Use async engine with run_sync for both SQLite and PostgreSQL
        async with self.engine.begin() as conn:
            await conn.run_sync(self.base_entities.metadata.create_all, checkfirst=True)

        txt.print(">>> Database & tables creation completed successfully (async)")

    @asynccontextmanager
    async def new_transaction_async(self):
        transaction = self.local_session()
        try:
            yield transaction
            await transaction.commit()
        except Exception:
            await transaction.rollback()
            raise
        finally:
            await transaction.close()
    
    @asynccontextmanager
    async def read_db_async(self):
        async with self.local_session() as session:
            try:
                yield session
            except Exception as e:
                raise RuntimeError(f"Error during read operation: {e}")

    @asynccontextmanager
    async def get_session_async(self):
        """Alias for read_db_async for compatibility with user repositories."""
        async with self.read_db_async() as session:
            yield session

    # AVOID - As it doesn't automatically map ORM objects
    @asynccontextmanager
    async def low_level_db_async(self):
        async with self.engine.connect() as connection:
            try:
                yield connection
            except Exception as e:
                raise RuntimeError(f"Error during read operation: {e}")
            
    async def does_exist_entity_by_id_async(self, entity_class, entity_id) -> bool:
        id_from_db = await self.get_first_entity_async(entity_class, 
                                                filters=[entity_class.id == entity_id], 
                                                selected_columns=[entity_class.id], 
                                                fails_if_not_found=False) 
        return id_from_db is not None

    async def get_entity_by_id_async(self, entity_class, entity_id, selected_columns: Optional[List] = None, to_join_list: Optional[List] = None, fails_if_not_found=False) -> Any | None:
        filters = [entity_class.id == entity_id]
        return await self.get_first_entity_async(entity_class=entity_class, filters=filters, selected_columns=selected_columns, to_join_list=to_join_list, fails_if_not_found=fails_if_not_found)
    
    async def get_first_entity_async(self, entity_class, filters: Optional[List[BinaryExpression]] = None, selected_columns: Optional[List] = None, to_join_list: Optional[List] = None, fails_if_not_found: bool = True) -> Any | None:
        query = select(*selected_columns) if selected_columns else select(entity_class)
        
        if filters:
            for filter_condition in filters:
                query = query.filter(filter_condition)

        if to_join_list:
            query = query.options(*[joinedload(to_join) for to_join in to_join_list])

        async with self.read_db_async() as session:
            try:
                results = await session.execute(query)
                if selected_columns:
                    # If only one column selected, return scalar
                    if len(selected_columns) == 1:
                        result = results.scalars().first()
                    else:
                        # Multiple columns selected
                        row = results.first()
                        if row is not None:
                            result = row
                        else:
                            result = None
                else:
                    # If no columns are selected, return the entire entity
                    result = results.scalars().first()

                if fails_if_not_found and not result:
                    filters_str = "".join([str(filter) for filter in filters]) if filters else ""
                    raise ValueError(f"No entity found for '{entity_class.__name__}' with filters: '{filters_str}'.")
                return result
            except Exception as e:
                filters_str = "".join([str(filter) for filter in filters]) if filters else ""
                txt.print(f'/!\\ Fails to retrieve first entity with filters: "{filters_str}" - Error: {e}')
                raise

    async def get_all_entities_async(self, entity_class, filters: Optional[List[BinaryExpression]] = None) -> list:
        query = select(entity_class)
        if filters:
            for filter_condition in filters:
                query = query.filter(filter_condition)

        async with self.read_db_async() as session:
            try:
                results = await session.execute(query)
                return results.unique().scalars().all()
            except Exception as e:
                txt.print(f"/!\\ Fails to retrieve entities: {e}")
                raise
            
    async def count_entities_async(self, entity_class, filters: Optional[List[BinaryExpression]] = None) -> int:
        query = select(func.count())
        if filters:
            for filter_condition in filters:
                query = query.where(filter_condition)

        async with self.read_db_async() as session:
            try:
                results = await session.execute(query)
                return results.scalar()
            except Exception as e:
                txt.print(f"/!\\ Fails to count entities: {e}")
                return 0

    async def add_entity_async(self, entity) -> any:
        results = await self.add_entities_async(entity)
        return results[0]
    
    async def add_entities_async(self, *args) -> list:
        async with self.new_transaction_async() as transaction:
            try:
                # Add all entities provided in *args
                for entity in args:
                    transaction.add(entity)
                await transaction.commit()
                return list(args)
            
            except Exception as e:
                txt.print(f"/!\\ Fails to add entities: {e}")
                raise

    async def update_entity_async(self, entity_class, entity_id, **kwargs) -> None:
        async with self.new_transaction_async() as transaction:
            try:
                result = await transaction.execute(select(entity_class).filter(entity_class.id == entity_id))
                entity = result.scalars().first()
                if not entity: 
                    raise ValueError(f"{entity_class.__name__} with id: {str(entity_id)} not found")

                for key, value in kwargs.items():
                    if hasattr(entity, key):
                        setattr(entity, key, value)

                transaction.add(entity)
                await transaction.commit()
            except Exception as e:
                txt.print(f"/!\\ Fails to update entity: {e}")
                raise

    async def delete_entity_async(self, entity_class, entity_id) -> None:
        async with self.new_transaction_async() as transaction:
            try:
                result = await transaction.execute(select(entity_class).filter(entity_class.id == entity_id))
                entity = result.scalars().first()
                if not entity:
                    raise ValueError(f"{entity_class.__name__} not found")
                
                await transaction.delete(entity)    
                await transaction.commit()
            except Exception as e:
                txt.print(f"/!\\ Fails to delete entity: {e}")
                raise

    async def empty_all_database_tables_async(self) -> None:
        async with self.new_transaction_async() as transaction:
            # Delete all tables records, tables in the reverse order to avoid foreign key integrity errors
            for table in reversed(self.base_entities.metadata.sorted_tables):
                await transaction.execute(delete(table))
            await transaction.commit()