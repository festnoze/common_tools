import asyncio
import uuid
import os
import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from sqlalchemy import Column, String
from sqlalchemy.orm import declarative_base
#
from common_tools.database.generic_datacontext import GenericDataContext
from common_tools.helpers.txt_helper import txt

pytest_plugins = ["pytest_asyncio"]

Base = declarative_base()

# Sample Entity for Testing
class SampleEntity(Base):
    __tablename__ = "sample_entities"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)


class TestGenericDataContext:
    def setup_method(self):
        # Create a temporary directory for the test database
        self.test_dir = Path(tempfile.gettempdir())
        
        # Use an absolute path for the database file
        self.db_path = self.test_dir / f"test_db_{uuid.uuid4()}.db"
        self._remove_db_file()
        self.data_context = GenericDataContext(Base, db_path_or_url=str(self.db_path))
        asyncio.run(self.data_context.empty_all_database_tables_async())

    def teardown_method(self):
        asyncio.run(self.data_context.empty_all_database_tables_async())
        self.data_context.engine.dispose()
        self._remove_db_file()

    ### TESTS ###

    @pytest.mark.asyncio
    async def test_add_entity(self):
        entity = SampleEntity(id=str(uuid4()), name="Test Entity")
        await self.data_context.add_entity_async(entity)

        result = await self.data_context.get_entity_by_id_async(SampleEntity, entity.id)
        assert result is not None
        assert result.id == entity.id
        assert result.name == entity.name

    @pytest.mark.asyncio
    async def test_get_entity_by_id(self):
        entity = SampleEntity(id=str(uuid4()), name="Test Entity")
        await self.data_context.add_entity_async(entity)

        result = await self.data_context.get_entity_by_id_async(SampleEntity, entity.id)
        assert result is not None
        assert result.id == entity.id

    @pytest.mark.asyncio
    async def test_get_all_entities(self):
        entities = [
            SampleEntity(id=str(uuid4()), name=f"Entity {i}") for i in range(3)
        ]
        for entity in entities:
            await self.data_context.add_entity_async(entity)

        results = await self.data_context.get_all_entities_async(SampleEntity)
        assert len(results) == len(entities)

    @pytest.mark.asyncio
    async def test_update_entity(self):
        entity = SampleEntity(id=str(uuid4()), name="Original Name")
        await self.data_context.add_entity_async(entity)

        await self.data_context.update_entity_async(
            SampleEntity, entity.id, name="Updated Name"
        )
        updated_entity = await self.data_context.get_entity_by_id_async(
            SampleEntity, entity.id
        )
        assert updated_entity.name == "Updated Name"

    @pytest.mark.asyncio
    async def test_delete_entity(self):
        entity = SampleEntity(id=str(uuid4()), name="Test Entity")
        await self.data_context.add_entity_async(entity)

        await self.data_context.delete_entity_async(SampleEntity, entity.id)
        result = await self.data_context.get_entity_by_id_async(
            SampleEntity, entity.id, fails_if_not_found=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_all_database_tables(self):
        entities = [
            SampleEntity(id=str(uuid4()), name=f"Entity {i}") for i in range(5)
        ]
        for entity in entities:
            await self.data_context.add_entity_async(entity)

        await self.data_context.empty_all_database_tables_async()
        results = await self.data_context.get_all_entities_async(SampleEntity)
        assert len(results) == 0

    def _remove_db_file(self):
        try:
            if self.db_path.exists():
                os.remove(self.db_path)
        except Exception as e:
            txt.print(f"/!\\ Fails to remove test DB file: {e}")