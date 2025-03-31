import asyncio
import uuid
import pytest
from langchain_core.documents import Document
from common_tools.workflows.workflow_output_decorator import workflow_output
from common_tools.workflows.workflow_executor import WorkflowExecutor

class TestWorkflowExecutor:
    def setup_method(self, method):
        self.config = {}
        self.executor = WorkflowExecutor(config_or_config_file_path=self.config, available_classes={'workflow_executor_test_methods': workflow_executor_test_methods})

    def teardown_method(self, method):
        self.config = None
        self.executor = None

    @pytest.mark.asyncio
    async def test_execute_workflow_with_both_config_and_entry_point(self):
        workflow_config = {
            'entry_point': ['workflow_executor_test_methods.step_one_async']
        }
        kwargs = {'a': 2, 'b': 3}
        results = await self.executor.execute_workflow_async(workflow_config, kwargs_values=kwargs, config_entry_point_name='entry_point')
        assert results == [5]

    @pytest.mark.asyncio
    async def test_execute_workflow_no_config_no_entry_point(self):
        with pytest.raises(ValueError, match='Starting step must either be provided or a step named "start" must be set in config.'):
            await self.executor.execute_workflow_async()

    @pytest.mark.asyncio
    async def test_execute_step_with_unexpected_type(self):
        step = 123
        with pytest.raises(TypeError, match='Invalid step type: int'):
            await self.executor.execute_step_async(step, [], {}, {})

    @pytest.mark.asyncio
    async def test_execute_dict_step_with_sub_workflow(self):
        step = {'sub_workflow': ['workflow_executor_test_methods.step_two_async']}
        self.executor.config = {'sub_workflow': ['workflow_executor_test_methods.step_two_async']}
        results = await self.executor.execute_dict_step_async(step, [5], {})
        assert results == [10]

    @pytest.mark.asyncio
    async def test_execute_str_step_with_invalid_step(self):
        step = 'non_existent_step'
        workflow_config = {}
        with pytest.raises(ValueError, match="Invalid function name 'non_existent_step'. It should be in 'Class.method' format."):
            await self.executor.execute_str_step_async(step, [], {}, workflow_config)

    @pytest.mark.asyncio
    async def test_execute_str_step_with_invalid_class_function(self):
        step = 'fakeclass.non_existent_function'
        workflow_config = {}
        with pytest.raises(ValueError, match="Class 'fakeclass' not found."):
            await self.executor.execute_str_step_async(step, [], {}, workflow_config)

    @pytest.mark.asyncio
    async def test_execute_str_step_with_invalid_function(self):
        step = 'workflow_executor_test_methods.non_existent_function'
        workflow_config = {}
        with pytest.raises(AttributeError, match="Class 'workflow_executor_test_methods' does not have a callable method 'non_existent_function'."):
            await self.executor.execute_str_step_async(step, [], {}, workflow_config)

    @pytest.mark.asyncio
    async def test_execute_function_with_exception(self):
        with pytest.raises(RuntimeError, match='Test exception'):
            await self.executor.execute_function_async('workflow_executor_test_methods.faulty_function', [], {})

    @pytest.mark.asyncio
    async def test_execute_function_multiple_workflow_outputs(self):
        kwargs = {'a': 2, 'b': 3}
        await self.executor.execute_function_async('workflow_executor_test_methods.step_four_w_2_workflow_outputs_async', [], kwargs)
        assert kwargs['sum'] == 5
        assert kwargs['product'] == 6

    @pytest.mark.asyncio
    async def test_execute_function_with_less_outputs_than_awaited_workflow_outputs_failed(self):
        kwargs = {'a': 2, 'b': 3}
        with pytest.raises(RuntimeError, match='Function only returned 1 values, but at least 2 were expected to match with output names decorator.'):
            await self.executor.execute_function_async('workflow_executor_test_methods.wrong_step_four_w_2_workflow_outputs_async', [], kwargs)

    @pytest.mark.asyncio
    async def test_execute_function_with_more_outputs_than_awaited_workflow_outputs_succeed(self):
        kwargs = {'a': 2, 'b': 3}
        await self.executor.execute_function_async('workflow_executor_test_methods.step_five_w_2_workflow_outputs_and_3_outputs_async', [], kwargs)
        assert kwargs['sum'] == 5
        assert kwargs['product'] == 6

    @pytest.mark.asyncio
    async def test_execute_workflow_kwargs(self):
        steps_config = ['workflow_executor_test_methods.step_two_async', 'workflow_executor_test_methods.step_three_async']
        kwargs = {'c': 10}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [19] # 10 * 2 - 1 = 19

    @pytest.mark.asyncio
    async def test_update_kwargs_with_no_return_info(self):
        async def dummy_function(a):
            return a * 2
        kwargs = {'a': 3}
        self.executor._add_function_output_names_and_values_to_kwargs(dummy_function, 4, kwargs_values=kwargs)
        assert kwargs == {'a': 3}

    @pytest.mark.asyncio
    async def test_execute_function_async_with_kwargs_values(self):
        result = await self.executor.execute_function_async('workflow_executor_test_methods.step_two_async', [], {'c': 3})
        assert result == 6

    @pytest.mark.asyncio
    async def test_execute_workflow_empty_config(self):
        results = await self.executor.execute_workflow_async([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_workflow_with_invalid_step_in_list(self):
        steps_config = ['workflow_executor_test_methods.step_one_async', 123]
        kwargs = {'a': 2, 'b': 3}
        with pytest.raises(TypeError, match='Invalid step type: int'):
            await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)

    @pytest.mark.asyncio
    async def test_execute_workflow_recursive_sub_workflows(self):
        self.executor.config = {'sub_workflow': ['sub_workflow']}
        steps_config = ['sub_workflow']
        with pytest.raises(RecursionError):
            await self.executor.execute_workflow_async(steps_config)

    def test_get_required_args(self):
        def func(a, b, c=3):
            pass
        provided_kwargs = {'a': 1, 'b': 2, 'd': 4}
        expected = {'a': 1, 'b': 2}

        result = self.executor._prepare_arguments_for_function(func, [], provided_kwargs)
        assert result == expected

    @pytest.mark.parametrize("nested_list, expected", [
        ([1, (2, (3, 4)), 5, {6, 7}], [1, 2, 3, 4, 5, 6, 7]), # Basic nested list with tuples and sets, including different levels of nesting
        ([1, {2, 3}, (4, 5)], [1, 2, 3, 4, 5]), # Set containing nested tuples
        ([None, (1, 2)], [None, 1, 2]), # Mix of None and nested tuples
        ([1, (2, 3), {4, 5}], [1, 2, 3, 4, 5]), # Nested tuples and sets
        (['a', ('b', ('c', 'd'))], ['a', 'b', 'c', 'd']), # Strings should remain intact
        ([1, (2, {"key": "value"})], [1, 2, {"key": "value"}]), # Dictionary within list - dicts shouldn't be flattened
        ([1, ('a', {True, 3.14}), 8], [1, 'a', True, 3.14, 8]), # Mixed data types with tuples and sets
        ([1, 2, 3, 4], [1, 2, 3, 4]), # Completely flat list - should return as is
        ([1, (2, (3, {4, 5})), 6], [1, 2, 3, 4, 5, 6]), # Deeply nested with tuples and sets
        ([1, ((), set(), (2, 3)), {}, []], [1, 2, 3, {}, []]), # Empty sets or tuples removed, not empty list or dict
        ([1, ['a', [True, [3.14, None]]], {'key': 'value'}], [1, ['a', [True, [3.14, None]]], {'key': 'value'}]), # Nested lists of lists
        ([1, (2, {'a': 10, 'b': 20}), 3], [1, 2, {'a': 10, 'b': 20}, 3]), # List containing dictionary - dicts shouldn't be flattened
    ])
    def test_flatten(self, nested_list, expected):
        result = self.executor.flatten_tuples(nested_list)
        assert result == expected

    def test_prepare_arguments_simple_case(self):
        func = workflow_executor_test_methods.sample_function_async
        previous_results = [42, "test", 3.14]
        kwargs_value = {}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 42, 'arg2': 'test', 'arg3': 3.14}

    def test_prepare_arguments_mixed_case(self):
        func = workflow_executor_test_methods.sample_function_async
        previous_results = ["test", 3.14]
        kwargs_value = {'arg1': 42, 'arg3': 2.71}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 42, 'arg2': 'test', 'arg3': 2.71}

    def test_prepare_arguments_complex_case(self):
        func = workflow_executor_test_methods.sample_function_async
        previous_results = ["from_previous_results", 123]
        kwargs_value = {'arg1': 1, 'arg3': 3.14}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 1, 'arg2': 'from_previous_results', 'arg3': 3.14}

    def test_missing_required_argument(self):
        func = workflow_executor_test_methods.sample_function_async
        previous_results = []
        kwargs_value = {}
        with pytest.raises(TypeError, match="Missing argument: 'arg1', which is required, because it has no default value."):
            self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)

    def test_prepare_arguments_with_defaults(self):
        func = workflow_executor_test_methods.another_function_async
        previous_results = [5]
        kwargs_value = {}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 5}

    def test_prepare_arguments_varargs_case(self):
        func = workflow_executor_test_methods.varargs_function_async
        previous_results = [10, "extra", 2.5]
        kwargs_value = {}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 10, 'arg2': 'extra', 'arg3': 2.5}
        assert all(previous_results)

    def test_kwarg_override(self):
        func = workflow_executor_test_methods.sample_function_async
        previous_results = [42, "test"]
        kwargs_value = {'arg3': 6.28, 'arg4': False}
        kwargs = self.executor._prepare_arguments_for_function(func, previous_results, kwargs_value)
        assert kwargs == {'arg1': 42, 'arg2': 'test', 'arg3': 6.28, 'arg4': False}

    @pytest.mark.asyncio
    async def test_execute_from_kwargs(self):
        kwargs = {'a': 1, 'b': 2}
        result = await self.executor.execute_function_async('workflow_executor_test_methods.step_one_async', [], kwargs)
        assert result == 3

    @pytest.mark.asyncio
    async def test_execute_from_kwargs_and_previous_results(self):
        kwargs = {'a': 1}
        previous_results = [51]
        result = await self.executor.execute_function_async('workflow_executor_test_methods.step_one_async', previous_results, kwargs)
        assert result == 52

    @pytest.mark.asyncio
    async def test_execute_step_subsequent_step(self):
        previous_results = [3]
        result = await self.executor.execute_function_async('workflow_executor_test_methods.step_two_async', previous_results, {})
        assert result == 6

    @pytest.mark.asyncio
    async def test_execute_workflow_single_step(self):
        steps_config = ['workflow_executor_test_methods.step_one_async']
        kwargs = {'a': 2, 'b': 3}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [5]

    @pytest.mark.asyncio
    async def test_execute_workflow_sequential_steps(self):
        steps_config = ['workflow_executor_test_methods.step_one_async', 'workflow_executor_test_methods.step_two_async', 'workflow_executor_test_methods.step_three_async']
        kwargs = {'a': 2, 'b': 3}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [9]

    @pytest.mark.asyncio
    async def test_execute_workflow_parallel_async(self):
        steps_config = [{'parallel_async': ['workflow_executor_test_methods.step_async', 'workflow_executor_test_methods.step_three_async']}]
        kwargs = {'e': 3, 'd': 10}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [9, 9]

    @pytest.mark.asyncio
    async def test_execute_workflow_nested_steps(self):
        self.executor.config = {'nested_steps': ['workflow_executor_test_methods.step_two_async', 'workflow_executor_test_methods.step_three_async']}
        steps_config = ['workflow_executor_test_methods.step_one_async', 'nested_steps']
        kwargs = {'a': 2, 'b': 3}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [9]

    @pytest.mark.asyncio
    async def test_execute_parallel_async(self):
        self.executor.config = {'start': {'parallel_async': ['workflow_executor_test_methods.step_async', 'workflow_executor_test_methods.step_three_async']}}
        kwargs = {'d': 5, 'e': 4}
        results = await self.executor.execute_workflow_async(kwargs_values=kwargs)
        assert results == [16, 4]

    @pytest.mark.asyncio
    async def test_execute_parallel_async_sub_workflow(self):
        self_config = {'sub1': {'parallel_async': ['workflow_executor_test_methods.step_async', 'workflow_executor_test_methods.step_three_async']}}
        kwargs = {'d': 5, 'e': 4}
        results = await self.executor.execute_workflow_async(self_config, kwargs_values=kwargs, config_entry_point_name='sub1')
        assert results == [16, 4]

    @pytest.mark.asyncio
    async def test_execute_step_missing_args(self):
        with pytest.raises(TypeError):
            await self.executor.execute_function_async('workflow_executor_test_methods.step_one_async', [], {})

    @pytest.mark.asyncio
    async def test_execute_workflow_missing_function(self):
        steps_config = ['unknown_step']
        with pytest.raises(ValueError):
            await self.executor.execute_workflow_async(steps_config, {})

    @pytest.mark.asyncio
    async def test_execute_workflow_full(self):
        self.executor.config = {'sub_workflow': [{'parallel_async': ['workflow_executor_test_methods.step_two_async', 'workflow_executor_test_methods.step_three_async']}]}
        steps_config = ['workflow_executor_test_methods.step_one_async', 'sub_workflow']
        kwargs = {'a': 2, 'b': 3, 'c': 4, 'd': 6}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert results == [8, 5]

    @pytest.mark.asyncio
    async def test_invalid_workflow_config(self):
        steps_config = 123
        with pytest.raises(TypeError):
            await self.executor.execute_workflow_async(steps_config)

    @pytest.mark.asyncio
    async def test_execute_workflow_no_kwargs(self):
        steps_config = ['workflow_executor_test_methods.step_four_async']
        result = await self.executor.execute_workflow_async(steps_config)
        assert result == [7]

    @pytest.mark.asyncio
    async def test_execute_function_using_previous_results(self):
        previous_results = [1, 2]
        result = await self.executor.execute_function_async('workflow_executor_test_methods.step_one_async', previous_results, {})
        assert result == 3

    @pytest.mark.asyncio
    async def test_parallel_async_with_empty_steps(self):
        results = await self.executor.execute_workflow_async([], [], {})
        assert results == []

    def test_execute_function_invalid_class(self):
        with pytest.raises(ValueError):
            self.executor.execute_function('InvalidClass.method', [], {})

    def test_execute_function_invalid_method(self):
        with pytest.raises(AttributeError):
            self.executor.execute_function('workflow_executor_test_methods.invalid_method', [], {})
    
    @pytest.mark.asyncio
    async def test_execute_function_async_invalid_class(self):
        with pytest.raises(ValueError):
            await self.executor.execute_function_async('InvalidClass.method', [], {})

    @pytest.mark.asyncio
    async def test_execute_function_async_invalid_method(self):
        with pytest.raises(AttributeError):
            await self.executor.execute_function_async('workflow_executor_test_methods.invalid_method', [], {})

    @pytest.mark.asyncio
    async def test_nested_parallel_async(self):
        self.executor.config = {
            'start': {'parallel_async': ['workflow_executor_test_methods.step_two_async', 'nested_parallel']},
            'nested_parallel': {'parallel_async': ['workflow_executor_test_methods.step_async', 'workflow_executor_test_methods.step_three_async']}
        }
        kwargs = {'c': 3, 'd': 4, 'e': 2}
        results = await self.executor.execute_workflow_async(kwargs_values=kwargs)
        assert results == [6, [4, 3]]

    def test_get_function_kwargs_missing_required_arg(self):
        provided_kwargs = {'x': 1}
        with pytest.raises(TypeError, match="Missing argument: 'y', which is required, because it has no default value."):
            self.executor._prepare_arguments_for_function(workflow_executor_test_methods.step_five_async, [], provided_kwargs)

    def test_get_function_kwargs_wrong_arg_name(self):
        provided_kwargs = {'x': 1, 'a': 2}
        with pytest.raises(TypeError, match="Missing argument: 'y', which is required, because it has no default value."):
            self.executor._prepare_arguments_for_function(workflow_executor_test_methods.step_five_async, [], provided_kwargs)

    def test_get_function_kwargs_having_default_value(self):
        provided_kwargs = {'x': 1, 'y': 2}
        prepared_args = self.executor._prepare_arguments_for_function(workflow_executor_test_methods.step_five_async, [], provided_kwargs)
        assert prepared_args == {'x': 1, 'y': 2}

    def test_get_function_kwargs_extra_provided(self):
        provided_kwargs = {'x': 1, 'y': 2, 'z': 3, 'a': 4}
        prepared_args = self.executor._prepare_arguments_for_function(workflow_executor_test_methods.step_five_async, [], provided_kwargs)
        assert prepared_args == {'x': 1, 'y': 2, 'z': 3}

    def test_flatten_invalid_input(self):
        invalid_input = 123
        with pytest.raises(TypeError):
            self.executor.flatten_tuples(invalid_input)

    # @pytest.mark.parametrize("data, expected", [
    #     ([(Document(page_content=f"content_{uuid.uuid4()}", metadata={}), 0.5), (Document(page_content=f"content_{uuid.uuid4()}", metadata={}), 1.0)], True),
    #     ([(Document(page_content=f"content_{uuid.uuid4()}", metadata={}), 0.5), (1.5, 2.0)], False)
    # ])
    # def test_type_matching_list_of_tuples_document_float(self, data, expected):
    #     param_annotation = list[tuple[Document, float]]
    #     assert self.executor._is_value_matching_annotation(data, param_annotation) == expected

    @pytest.mark.parametrize("data, expected", [
        ((1, 2), True),
        ((1, "two"), False)
    ])
    def test_type_matching_tuple_of_ints(self, data, expected):
        param_annotation = tuple[int, int]
        assert self.executor._is_value_matching_annotation(data, param_annotation) == expected

    @pytest.mark.parametrize("data, expected", [
        ({"one": 1, "two": 2}, True),
        ({1: "one", 2: "two"}, False)
    ])
    def test_type_matching_dict_str_int(self, data, expected):
        param_annotation = dict[str, int]
        assert self.executor._is_value_matching_annotation(data, param_annotation) == expected

    def test_type_matching_list_of_documents(self):
        documents = [Document(page_content=f"content_{uuid.uuid4()}", metadata={}), Document(page_content=f"content_{uuid.uuid4()}", metadata={})]
        param_annotation = list[Document]
        assert self.executor._is_value_matching_annotation(documents, param_annotation)

    # def test_type_matching_none_value(self):
    #     param_annotation = list[Document]
    #     assert self.executor._is_value_matching_annotation(None, param_annotation)

    def test_type_matching_empty_list(self):
        param_annotation = list[Document]
        assert self.executor._is_value_matching_annotation([], param_annotation)

    def test_get_function_ouptut_names_with_single_output(self):
        @workflow_output('result')
        async def dummy_function():
            pass
        return_info = self.executor._get_function_workflow_outputs(dummy_function)
        assert return_info == {'workflow_outputs': ['result']}

    def test__function_ouptut_names_with_multiple_outputs(self):
        @workflow_output('result1', 'result2')
        async def dummy_function():
            pass
        return_info = self.executor._get_function_workflow_outputs(dummy_function)
        assert return_info == {'workflow_outputs': ['result1', 'result2']}

    @pytest.mark.asyncio
    async def test_execute_function_single_output(self):
        async def dummy_function(a, b):
            return a + b
        dummy_function._workflow_output = 'sum'
        self.executor.get_static_method = lambda x: dummy_function
        kwargs = {'a': 2, 'b': 3}
        await self.executor.execute_function_async('dummy_function', [], kwargs)
        assert kwargs['sum'] == 5

    @pytest.mark.asyncio
    async def test_execute_function_multiple_outputs(self):
        async def dummy_function(a, b):
            return a + b, a * b
        dummy_function._workflow_output = ['sum', 'product']
        self.executor.get_static_method = lambda x: dummy_function
        kwargs = {'a': 2, 'b': 3}
        await self.executor.execute_function_async('dummy_function', [], kwargs)
        assert kwargs['sum'] == 5
        assert kwargs['product'] == 6

    @pytest.mark.asyncio
    async def test_execute_workflow_with_named_outputs(self):
        self.executor.get_static_method = lambda x: workflow_executor_test_methods.step_one_w_workflow_output_async if x == 'step_one' else workflow_executor_test_methods.step_two_w_workflow_output_async
        steps_config = ['step_one', 'step_two']
        kwargs = {'a': 2, 'b': 3}
        results = await self.executor.execute_workflow_async(steps_config, kwargs_values=kwargs)
        assert kwargs['sum'] == 5
        assert kwargs['double'] == 10
        assert results == [10]

    @pytest.mark.parametrize("kwargs_value, previous_results, expected_kwargs", [
        ({'x': 1}, [1, 2, 3], {'x': 1, 'y': 2, 'z': 3}),
        ({'y': 3}, [2, 3, 4], {'x': 2, 'y': 3, 'z': 4}),
    ])
    def test_prepare_arguments_for_function(self, kwargs_value, previous_results, expected_kwargs):
        prepared_kwargs = self.executor._prepare_arguments_for_function(workflow_executor_test_methods.step_five_async, previous_results, kwargs_value)
        assert prepared_kwargs == expected_kwargs, f"Expected {expected_kwargs} but got {prepared_kwargs}"

class workflow_executor_test_methods:
    async def step_one_async(a, b):
        return a + b

    async def step_two_async(c):
        return c * 2

    async def step_three_async(d):
        return d - 1

    async def step_four_async():
        return 7

    async def step_five_async(x, y, z=3):
        return x + y + z
    
    def faulty_function():
            raise RuntimeError('Test exception')

    @workflow_output('sum')
    async def step_one_w_workflow_output_async(a, b):
        return a + b

    @workflow_output('double')
    async def step_two_w_workflow_output_async(c):
        return c * 2

    @workflow_output('product')
    async def step_three_using_workflow_outputs_async(sum, workflow_output):
        return sum * workflow_output

    @workflow_output('sum', 'product')
    async def step_four_w_2_workflow_outputs_async(a, b):
        return a + b, a * b

    @workflow_output('sum', 'product')
    async def wrong_step_four_w_2_workflow_outputs_async(a, b):
        return a + b

    @workflow_output('sum', 'product')
    async def step_five_w_2_workflow_outputs_and_3_outputs_async(a, b):
        return a + b, a * b, a - b

    async def step_async(e):
        await asyncio.sleep(0.1)
        return e ** 2

    async def sample_function_async(arg1: int, arg2: str, arg3: float, arg4: bool = True):
        pass

    async def another_function_async(arg1: int, arg2: str = "default", arg3: list = None):
        pass

    async def varargs_function_async(arg1: int, *args, arg2: str = "default", arg3: float = 1.0):
        pass
