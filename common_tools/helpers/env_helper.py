# /!\ 'load_dotenv()'  Must be done beforehand in the main script!
import json
import os
from dotenv import load_dotenv
#
from common_tools.helpers.txt_helper import txt
from common_tools.helpers.file_helper import file
from common_tools.models.llm_info import LlmInfo
from common_tools.models.vector_db_type import VectorDbType
from common_tools.models.embedding_model import EmbeddingModel
from common_tools.models.langchain_adapter_type import LangChainAdapterType

class EnvHelper:
    is_env_loaded = False

    @staticmethod
    def _get_custom_env_files():
        return EnvHelper.get_env_variable_value_by_name('CUSTOM_ENV_FILES', load_env=False, fails_if_missing=False)
    
    @staticmethod
    def get_openai_api_key():
        return EnvHelper.get_env_variable_value_by_name('OPENAI_API_KEY')
    
    @staticmethod
    def get_anthropic_api_key():
        return EnvHelper.get_env_variable_value_by_name('ANTHROPIC_API_KEY')
    
    @staticmethod
    def get_groq_api_key():
        return EnvHelper.get_env_variable_value_by_name('GROQ_API_KEY')

    @staticmethod
    def get_pinecone_api_key():
        return EnvHelper.get_env_variable_value_by_name('PINECONE_API_KEY')

    @staticmethod
    def get_pinecone_environment():
        return EnvHelper.get_env_variable_value_by_name('PINECONE_ENVIRONMENT')
    
    @staticmethod
    def get_openrouter_api_key():
        return EnvHelper.get_env_variable_value_by_name('OPENROUTER_API_KEY')

    @staticmethod
    def get_openrouter_base_url():
        return EnvHelper.get_env_variable_value_by_name('OPENROUTER_BASE_URL')
    
    @staticmethod
    def get_embedding_model() -> EmbeddingModel:
        embedding_model_value = EnvHelper.get_env_variable_value_by_name('EMBEDDING_MODEL')
        try:
            embedding_model = EmbeddingModel[embedding_model_value]
        except KeyError:
            raise ValueError(f"Invalid value for 'EMBEDDING_MODEL': '{embedding_model_value}' (cannot be found within 'EmbeddingModel' allowed values)")
        return embedding_model
    
    def get_embedding_size() -> int:
        embedding_size_str = EnvHelper.get_env_variable_value_by_name('EMBEDDING_SIZE')
        try:
            embedding_size = int(embedding_size_str) if embedding_size_str else 0
        except ValueError:
            raise ValueError(f"Invalid value for 'EMBEDDING_SIZE': '{embedding_size_str}' (cannot be converted to an integer)")
        return embedding_size
    
    @staticmethod
    def get_vector_db_type() -> VectorDbType:
        vector_db_type_str = EnvHelper.get_env_variable_value_by_name('VECTOR_DB_TYPE')
        try:
            vector_db_type = VectorDbType(vector_db_type_str)
        except KeyError:
            raise ValueError(f"Invalid value for 'VECTOR_DB_TYPE': '{vector_db_type_str}' (cannot be found within 'VectorDbType' allowed values)")
        return vector_db_type

    @staticmethod
    def get_vector_db_name():
        return EnvHelper.get_env_variable_value_by_name('VECTOR_DB_NAME')
    
    @staticmethod
    def get_BM25_storage_as_db_sparse_vectors() -> bool:
        var_name = 'BM25_STORAGE_AS_DB_SPARSE_VECTORS'
        str_value = EnvHelper.get_env_variable_value_by_name(var_name)
        return txt.get_bool_value_out_of_str_value(str_value, var_name)

    @staticmethod
    def get_is_common_db_for_sparse_and_dense_vectors() -> bool:
        var_name = 'IS_COMMON_DB_FOR_SPARSE_AND_DENSE_VECTORS'
        str_value = EnvHelper.get_env_variable_value_by_name(var_name)
        return txt.get_bool_value_out_of_str_value(str_value, var_name)
    
    @staticmethod
    def get_is_summarized_data() -> bool:
        var_name = 'IS_SUMMARIZED_DATA'
        str_value = EnvHelper.get_env_variable_value_by_name(var_name)
        return txt.get_bool_value_out_of_str_value(str_value, var_name) 
    
    @staticmethod
    def get_is_questions_created_from_data(fails_if_unfound_value = True) -> bool:
        var_name = 'IS_QUESTIONS_CREATED_FROM_DATA'
        str_value = EnvHelper.get_env_variable_value_by_name(var_name)
        return txt.get_bool_value_out_of_str_value(str_value, var_name, fails_if_unfound_value)
    
    @staticmethod
    def get_is_mixed_questions_and_data(fails_if_unfound_value = True) -> bool:
        var_name = 'IS_MIXED_QUESTIONS_AND_DATA'
        str_value = EnvHelper.get_env_variable_value_by_name(var_name)
        return txt.get_bool_value_out_of_str_value(str_value, var_name, fails_if_unfound_value)
    
    @staticmethod
    def get_llms_infos_from_env_config(skip_commented_lines:bool = True) -> list[LlmInfo]:
        yaml_env_variables = EnvHelper._get_llm_env_variables(skip_commented_lines)
        if not 'Llms_Temperature' in yaml_env_variables:
            raise ValueError("Missing 'Llms_Temperature' key in the yaml environment file")
        llms_temp = yaml_env_variables['Llms_Temperature']
        if not 'Llm_infos' in yaml_env_variables:
            raise ValueError("Missing 'Llm_infos' key in the yaml environment file")
        llms_list = yaml_env_variables['Llm_infos']
        if not 'Llms_order' in yaml_env_variables:
            raise ValueError("Missing 'Llms_order' key in the yaml environment file")
        llms_order = yaml_env_variables['Llms_order']

        # Re-order llms based on specified order
        try:
            if len(llms_order) > len(llms_list):
                llms_order = llms_order[:len(llms_list)]
            ordered_llms_list = [llms_list[i - 1] for i in llms_order]
            if len(llms_list) > len(ordered_llms_list):
                ordered_llms_list.extend([llm for llm in llms_list if llm not in ordered_llms_list])
        except IndexError:
            raise ValueError("The 'Llms_order' env variable list contains invalid indices.")

        try:
            llms_infos = []
            for llm_dict in ordered_llms_list:
                llm_info = LlmInfo.factory_from_dict(llm_dict, llms_temp)
                llms_infos.append(llm_info)
            return llms_infos
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing Llm_infos value: {e}")
        except KeyError as e:
            raise ValueError(f"Missing key in Llm_infos: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing Llm_infos value: {e}")

    @staticmethod
    def _get_llms_json() -> str:
        return EnvHelper.get_env_variable_value_by_name('LLMS_JSON')  
    
    @staticmethod
    def _init_load_env():
        if not EnvHelper.is_env_loaded:
            load_dotenv()
            EnvHelper._load_custom_env_files()
            EnvHelper.is_env_loaded = True

    @staticmethod
    def _load_custom_env_files():
        custom_env_files = EnvHelper._get_custom_env_files()
        
        # In case no custom additionnal env. files are defined into a 'CUSTOM_ENV_FILES' key of the '.env' file 
        if not custom_env_files:
            return 
        
        custom_env_filenames = [filename.strip() for filename in custom_env_files.split(",")]
        for custom_env_filename in custom_env_filenames:
            if not os.path.exists(custom_env_filename):
                raise FileNotFoundError(f"/!\\ Environment file: '{custom_env_filename}' was not found at the project root.")
            load_dotenv(custom_env_filename)
    
    @staticmethod
    def get_env_variable_value_by_name(variable_name: str, load_env=True, fails_if_missing=True) -> str:
        if variable_name not in os.environ:
            if load_env:
                EnvHelper._init_load_env()
            variable_value: str = os.getenv(variable_name)
            if not variable_value:
                if fails_if_missing:
                    raise ValueError(f'Variable named: "{variable_name}" is not set in the environment')
                else:
                    return None
            os.environ[variable_name] = variable_value            
        return os.environ[variable_name]
    
    def _get_llm_env_variables(skip_commented_lines:bool = True):
        return file.get_as_yaml('.llm.env.yaml', skip_commented_lines)
    