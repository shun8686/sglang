import json
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.server_args import PortArgs, prepare_server_args
from sglang.test.test_utils import CustomTestCase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.test_ascend_utils import META_LLAMA_3_1_8B_INSTRUCT
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestPrepareServerArgs(CustomTestCase):
    """Testcase：Verify the correctness of server startup argument parsing logic in prepare_server_args function

    [Test Category] Parameter
    [Test Target] --json-model-override-args
    """

    def test_prepare_server_args(self):
        """Test core functionality of prepare_server_args:
        1. Verify --model-path parameter parsing
        2. Verify --json-model-override-args JSON string parsing
        """
        # Parse server arguments with model path and JSON override args
        server_args = prepare_server_args(
            [
                "--model-path",
                META_LLAMA_3_1_8B_INSTRUCT
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )

        # Verify model path is parsed correctly
        expected_model_path = META_LLAMA_3_1_8B_INSTRUCT
        self.assertEqual(server_args.model_path, expected_model_path)

        # Verify JSON model override args are parsed correctly
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )


class TestPortArgs(unittest.TestCase):
    """Testcase：Verify the initialization logic and network address parsing correctness of PortArgs class

    [Test Category] Parameter
    [Test Target] --port;--nccl-port;--enable-dp-attention;--nnodes;--dist-init-addr;--tokenizer-worker-num
    """

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file, mock_is_port_available):
        """Test standard case (no DP attention):
        - Verify IPC names use ipc:// protocol
        - Verify NCCL port is generated as integer
        """

        mock_is_port_available.return_value = True
        mock_temp_file.return_value.name = "temp_file"

        # Create mock server arguments
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.tokenizer_worker_num = 1

        # Initialize PortArgs
        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))

        # Verify NCCL port is generated as integer
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_single_node_dp_attention(self, mock_is_port_available):
        """Test single node DP attention case:
        - Verify IPC names use tcp://127.0.0.1 protocol
        - Verify NCCL port is generated
        """
        mock_is_port_available.return_value = True

        # Mock server arguments for single node DP attention
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None
        server_args.tokenizer_worker_num = 1

        # Initialize PortArgs
        port_args = PortArgs.init_new(server_args)

        # Verify IPC names use TCP protocol with localhost
        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_dp_rank(self, mock_is_port_available):
        """Test DP rank configuration:
        - Verify scheduler input IPC name ends with correct rank port
        - Verify IP address from dist_init_addr is used
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with DP rank
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"
        server_args.tokenizer_worker_num = 1

        # Initialize PortArgs with DP rank and worker ports
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=[0, 1, 2])

        # Verify scheduler input port matches DP rank
        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":2"))

        # Verify IP address from dist_init_addr is used
        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_ipv4_address(self, mock_is_port_available):
        """Test valid IPv4 address in dist_init_addr:
        - Verify all IPC names use the specified IPv4 address
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with valid IPv4 address
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"
        server_args.tokenizer_worker_num = 1

        # Initialize PortArgs
        port_args = PortArgs.init_new(server_args)

        # Verify all IPC names use the specified IPv4 address
        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address(self, mock_is_port_available):
        """Test malformed IPv4 address (missing port):
        - Verify AssertionError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with malformed IPv4 (missing port)
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"
        server_args.tokenizer_worker_num = 1

        # Verify AssertionError is raised
        with self.assertRaises(AssertionError) as context:
            PortArgs.init_new(server_args)

        # Verify error message contains correct guidance
        self.assertIn(
            "please provide --dist-init-addr as host:port", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv4_address_invalid_port(
        self, mock_is_port_available
    ):
        """Test malformed IPv4 address (non-integer port):
        - Verify ValueError is raised
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with invalid port (string instead of integer)
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised for invalid port
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        """Test valid IPv6 address in dist_init_addr:
        - Verify all IPC names use the specified IPv6 address with correct format
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with valid IPv6 address
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:25000"
        server_args.tokenizer_worker_num = 1

        # Initialize PortArgs
        port_args = PortArgs.init_new(server_args)

        # Verify all IPC names use the specified IPv6 address with correct format
        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://[2001:db8::1]:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertTrue(
            port_args.detokenizer_ipc_name.startswith("tcp://[2001:db8::1]:")
        )
        self.assertIsInstance(port_args.nccl_port, int)

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=False)
    def test_init_new_with_invalid_ipv6_address(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        """Test invalid IPv6 address:
        - Verify ValueError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with invalid IPv6 address
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[invalid-ipv6]:25000"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        # Verify error message indicates invalid IPv6 address
        self.assertIn("invalid IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    def test_init_new_with_malformed_ipv6_address_missing_bracket(
        self, mock_is_port_available
    ):
        """Test malformed IPv6 address (missing closing bracket):
        - Verify ValueError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with malformed IPv6 (missing closing bracket)
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1:25000"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        # Verify error message indicates invalid IPv6 format
        self.assertIn("invalid IPv6 address format", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_missing_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        """Test malformed IPv6 address (missing port):
        - Verify ValueError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with malformed IPv6 (missing port)
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        # Verify error message indicates missing port
        self.assertIn(
            "a port must be specified in IPv6 address", str(context.exception)
        )

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_invalid_port(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        """Test malformed IPv6 address (non-integer port):
        - Verify ValueError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with invalid port in IPv6 address
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]:abcde"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        # Verify error message indicates invalid port
        self.assertIn("invalid port in IPv6 address", str(context.exception))

    @patch("sglang.srt.server_args.is_port_available")
    @patch("sglang.srt.server_args.is_valid_ipv6_address", return_value=True)
    def test_init_new_with_malformed_ipv6_address_wrong_separator(
        self, mock_is_valid_ipv6, mock_is_port_available
    ):
        """Test malformed IPv6 address (wrong separator between IP and port):
        - Verify ValueError is raised with correct message
        """
        mock_is_port_available.return_value = True

        # Mock server arguments with wrong separator (# instead of :)
        server_args = MagicMock()
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "[2001:db8::1]#25000"
        server_args.tokenizer_worker_num = 1

        # Verify ValueError is raised
        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        # Verify error message indicates expected colon separator
        self.assertIn("expected ':' after ']'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
