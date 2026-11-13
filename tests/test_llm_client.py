import unittest
from unittest.mock import patch

from llm import client


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class LLMClientTests(unittest.TestCase):
    def test_complete_json_handles_fenced_json_response(self):
        response = FakeResponse('```json\n{"info_complete": true, "clarifying_question": null}\n```')

        with patch('llm.client.litellm.completion', return_value=response) as completion:
            result = client.complete_json('system prompt', 'user content')

        self.assertEqual(result, {'info_complete': True, 'clarifying_question': None})
        self.assertEqual(completion.call_args.kwargs['num_retries'], 2)


if __name__ == '__main__':
    unittest.main()
