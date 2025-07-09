from common_tools.helpers.txt_helper import txt

class display:
    def display_assistant_ids(title, assistant_set):
        txt.print(f"{title}:")
        txt.print(f"• assistant id: '{assistant_set.assistant.id}'")
        txt.print(f"• thread id:    '{assistant_set.thread.id}'")
        txt.print(f"----------------------------------------------")

    def get_llm_infos(llm):
        model_name = getattr(llm, 'model_name', None)
        model = getattr(llm, 'model', None)

        if model_name:
            return f"• LLM model: '{model_name}' & name: '{llm.name}'"
        elif model:
            return f"• LLM model: '{model}' & name: '{llm.name}'"
        else:
            return f"• LLM model: '{llm.name}'"
        