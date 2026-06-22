# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for HookRouter."""

import asyncio
from typing import Any
from absl.testing import absltest
from google.antigravity.connections.local import localharness_pb2
from google.antigravity.connections.local.hook_router import HookRouter
from google.antigravity.hooks import hook_runner as h_runner
from google.antigravity.hooks import hooks


class HookRouterTest(absltest.TestCase):

  def test_handle_on_session_start(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_start
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_start_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_1",
          name="OnSessionStart",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_START,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_1")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_on_session_end(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.on_session_end
      async def my_hook():
        fired.set()

      hook_runner = h_runner.HookRunner(
          on_session_end_hooks=[my_hook],
      )

      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_end",
          name="OnSessionEnd",
          type=localharness_pb2.LIFECYCLE_HOOK_ON_SESSION_END,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_end")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_unknown_hook(self):

    async def _test():
      hook_runner = h_runner.HookRunner()
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)

      req = localharness_pb2.CallHookRequest(
          request_id="test_req_unknown",
          name="UnknownHook",
          type=localharness_pb2.LIFECYCLE_HOOK_UNSPECIFIED,
      )

      await router.handle(req)

      self.assertLen(sent_events, 1)
      self.assertTrue(sent_events[0].HasField("call_hook_response"))
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_req_unknown")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())

  def test_handle_pre_turn_allow(self):

    async def _test():
      fired = asyncio.Event()

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        fired.set()
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_pre_turn",
          name="PreTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_pre_turn")
      self.assertTrue(resp.HasField("pre_turn_result"))
      self.assertEqual(
          resp.pre_turn_result.decision,
          localharness_pb2.PreTurnResult.Decision.ALLOW,
      )

    asyncio.run(_test())

  def test_handle_pre_turn_multipart_content(self):

    async def _test():
      captured_input = []

      @hooks.pre_turn
      async def my_hook(prompt: Any):
        captured_input.append(prompt)
        return hooks.HookResult(allow=True)

      hook_runner = h_runner.HookRunner(pre_turn_hooks=[my_hook])
      router = HookRouter(hook_runner, lambda _: asyncio.sleep(0))
      req = localharness_pb2.CallHookRequest(
          request_id="test_multi",
          type=localharness_pb2.LIFECYCLE_HOOK_PRE_TURN,
          pre_turn_args=localharness_pb2.PreTurnArgs(
              user_input=localharness_pb2.UserInput(
                  parts=[
                      localharness_pb2.UserInput.Part(text="hello"),
                      localharness_pb2.UserInput.Part(text="world"),
                  ]
              )
          ),
      )

      await router.handle(req)

      self.assertLen(captured_input, 1)
      self.assertEqual(captured_input[0], ["hello", "world"])

    asyncio.run(_test())

  def test_handle_post_turn(self):

    async def _test():
      fired = asyncio.Event()
      received_data = []

      @hooks.post_turn
      async def my_hook(data: str):
        fired.set()
        received_data.append(data)

      hook_runner = h_runner.HookRunner(post_turn_hooks=[my_hook])
      sent_events = []

      async def mock_send(event: localharness_pb2.InputEvent):
        sent_events.append(event)

      router = HookRouter(hook_runner, mock_send)
      req = localharness_pb2.CallHookRequest(
          request_id="test_post_turn",
          name="PostTurn",
          type=localharness_pb2.LIFECYCLE_HOOK_POST_TURN,
          post_turn_args=localharness_pb2.PostTurnArgs(response_text="final"),
      )

      await router.handle(req)

      self.assertTrue(fired.is_set())
      self.assertEqual(received_data, ["final"])
      self.assertLen(sent_events, 1)
      resp = sent_events[0].call_hook_response
      self.assertEqual(resp.request_id, "test_post_turn")
      self.assertTrue(resp.HasField("empty_result"))

    asyncio.run(_test())


if __name__ == "__main__":
  absltest.main()
