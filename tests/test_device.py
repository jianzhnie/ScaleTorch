"""Tests for scaletorch.utils.device module."""
import unittest
from unittest.mock import patch

from scaletorch.utils.device import (
    get_current_device,
    get_device_count,
    get_device_type,
    get_dist_backend,
    get_dist_info,
    get_theoretical_flops,
    get_visible_devices_keyword,
    is_accelerator_available,
    is_bf16_supported,
)


class TestDeviceType(unittest.TestCase):
    """Test device type detection."""

    def setUp(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def tearDown(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=True)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    def test_npu_detected(self, mock_cuda, mock_npu):
        self.assertEqual(get_device_type(), 'npu')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=True)
    def test_cuda_detected(self, mock_cuda, mock_npu):
        self.assertEqual(get_device_type(), 'cuda')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_xpu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mlu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_musa_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mps_available', return_value=False)
    def test_cpu_fallback(self, *mocks):
        self.assertEqual(get_device_type(), 'cpu')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=True)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    def test_cache_works(self, mock_cuda, mock_npu):
        result1 = get_device_type()
        result2 = get_device_type()
        self.assertEqual(result1, result2)
        mock_npu.assert_called_once()


class TestDistBackend(unittest.TestCase):
    """Test distributed backend selection."""

    def setUp(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def tearDown(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=True)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    def test_hccl_for_npu(self, mock_cuda, mock_npu):
        self.assertEqual(get_dist_backend(), 'hccl')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=True)
    def test_nccl_for_cuda(self, mock_cuda, mock_npu):
        self.assertEqual(get_dist_backend(), 'nccl')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_xpu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mlu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_musa_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mps_available', return_value=False)
    def test_gloo_fallback(self, *mocks):
        self.assertEqual(get_dist_backend(), 'gloo')


class TestDistInfo(unittest.TestCase):
    """Test distributed info retrieval."""

    @patch.dict('os.environ', {'RANK': '3', 'WORLD_SIZE': '8', 'LOCAL_RANK': '1'})
    def test_from_env(self):
        rank, world_size, local_rank = get_dist_info()
        self.assertEqual(rank, 3)
        self.assertEqual(world_size, 8)
        self.assertEqual(local_rank, 1)

    @patch.dict('os.environ', {}, clear=True)
    def test_defaults(self):
        rank, world_size, local_rank = get_dist_info()
        self.assertEqual(rank, 0)
        self.assertEqual(world_size, 1)
        self.assertEqual(local_rank, 0)


class TestCurrentDevice(unittest.TestCase):
    """Test get_current_device."""

    def setUp(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def tearDown(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def test_use_cpu(self):
        import torch
        device = get_current_device(use_cpu=True)
        self.assertEqual(device, torch.device('cpu'))

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_xpu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mlu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_musa_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_mps_available', return_value=False)
    def test_cpu_when_no_accelerator(self, *mocks):
        import torch
        device = get_current_device()
        self.assertEqual(device, torch.device('cpu'))


class TestTheoreticalFlops(unittest.TestCase):
    """Test theoretical FLOPS calculation."""

    def setUp(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def tearDown(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=True)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    def test_npu_flops(self, mock_cuda, mock_npu):
        flops = get_theoretical_flops()
        self.assertEqual(flops, 320.0e12)

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=True)
    def test_cuda_flops(self, mock_cuda, mock_npu):
        flops = get_theoretical_flops()
        self.assertEqual(flops, 989.5e12)


class TestVisibleDevices(unittest.TestCase):
    """Test visible devices keyword."""

    def setUp(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    def tearDown(self):
        import scaletorch.utils.device as dev_mod
        dev_mod._device_type_cache = None

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=True)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=False)
    def test_npu_env_var(self, mock_cuda, mock_npu):
        self.assertEqual(get_visible_devices_keyword(), 'ASCEND_RT_VISIBLE_DEVICES')

    @patch('scaletorch.utils.device.is_torch_npu_available', return_value=False)
    @patch('scaletorch.utils.device.is_torch_cuda_available', return_value=True)
    def test_cuda_env_var(self, mock_cuda, mock_npu):
        self.assertEqual(get_visible_devices_keyword(), 'CUDA_VISIBLE_DEVICES')


if __name__ == '__main__':
    unittest.main()
