# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

from tflite_micro.tensorflow.lite.micro.compression import decode


class TestDecodeCommonMetadata(tf.test.TestCase):

  def testBasicSerialization(self):
    dcm = decode.DecodeCommonMetadata(decode_type=decode.DecodeType.LUT)
    result = dcm.to_bytes()

    # Should be exactly 16 bytes
    self.assertEqual(len(result), 16)

    # Byte 0: decode_type
    self.assertEqual(result[0], 0)

    # Byte 1: version (default 1)
    self.assertEqual(result[1], 1)

    # Bytes 2-3: reserved (should be zero)
    self.assertEqual(result[2], 0)
    self.assertEqual(result[3], 0)

    # Bytes 4-15: user_data (default all zeros)
    self.assertEqual(result[4:16], b'\x00' * 12)

  def testCustomVersion(self):
    dcm = decode.DecodeCommonMetadata(decode_type=1, version=2)
    result = dcm.to_bytes()

    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], 2)

  def testUserData(self):
    user_data = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c'
    dcm = decode.DecodeCommonMetadata(decode_type=0, user_data=user_data)
    result = dcm.to_bytes()

    self.assertEqual(result[4:16], user_data)

  def testUserDataPadding(self):
    # User data shorter than 12 bytes should be padded with zeros
    user_data = b'\x01\x02\x03'
    dcm = decode.DecodeCommonMetadata(decode_type=0, user_data=user_data)
    result = dcm.to_bytes()

    expected = b'\x01\x02\x03' + b'\x00' * 9
    self.assertEqual(result[4:16], expected)

  def testUserDataTruncation(self):
    # User data longer than 12 bytes should be truncated
    user_data = b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f'
    dcm = decode.DecodeCommonMetadata(decode_type=0, user_data=user_data)
    result = dcm.to_bytes()

    self.assertEqual(result[4:16], user_data[:12])

  def testDecodeTypeRange(self):
    # Valid decode types: 0-255
    decode.DecodeCommonMetadata(decode_type=decode.DecodeType.LUT).to_bytes()
    decode.DecodeCommonMetadata(decode_type=decode.DecodeType(127)).to_bytes()
    decode.DecodeCommonMetadata(
        decode_type=decode.DecodeType.custom(255)).to_bytes()

    # Invalid decode types should raise ValueError
    with self.assertRaises(ValueError):
      decode.DecodeCommonMetadata(decode_type=decode.DecodeType(-1)).to_bytes()
    with self.assertRaises(ValueError):
      decode.DecodeCommonMetadata(
          decode_type=decode.DecodeType(256)).to_bytes()

  def testVersionRange(self):
    # Valid versions: 0-255
    decode.DecodeCommonMetadata(decode_type=0, version=0).to_bytes()
    decode.DecodeCommonMetadata(decode_type=0, version=255).to_bytes()

    # Invalid versions should raise ValueError
    with self.assertRaises(ValueError):
      decode.DecodeCommonMetadata(decode_type=0, version=-1).to_bytes()
    with self.assertRaises(ValueError):
      decode.DecodeCommonMetadata(decode_type=0, version=256).to_bytes()


class TestAncillaryDataTensor(tf.test.TestCase):

  def testDcmOnly(self):
    dcm = decode.DecodeCommonMetadata(decode_type=decode.DecodeType.LUT)
    adt = decode.AncillaryDataTensor(dcm)
    result = adt.to_bytes()

    # Should be just the 16-byte DCM
    self.assertEqual(len(result), 16)
    self.assertEqual(result, dcm.to_bytes())

  def testWithBytesAncillaryData(self):
    dcm = decode.DecodeCommonMetadata(decode_type=decode.DecodeType.HUFFMAN)
    ancillary = b'\xaa\xbb\xcc\xdd'
    adt = decode.AncillaryDataTensor(dcm, ancillary)
    result = adt.to_bytes()

    # Should be DCM + ancillary data
    self.assertEqual(len(result), 20)
    self.assertEqual(result[:16], dcm.to_bytes())
    self.assertEqual(result[16:], ancillary)

  def testWithAncillaryDataMethod(self):
    dcm = decode.DecodeCommonMetadata(decode_type=decode.DecodeType.PRUNING)
    adt = decode.AncillaryDataTensor(dcm)

    ancillary = b'\x11\x22\x33\x44'
    adt_with_data = adt.with_ancillary_data(ancillary)
    result = adt_with_data.to_bytes()

    # Original ADT should be unchanged
    self.assertEqual(adt.to_bytes(), dcm.to_bytes())

    # New ADT should have ancillary data
    self.assertEqual(len(result), 20)
    self.assertEqual(result[:16], dcm.to_bytes())
    self.assertEqual(result[16:], ancillary)

  def testWithSerializerProtocol(self):
    # Test with an object that implements AncillaryDataSerializer
    class MockSerializer:

      def to_bytes(self):
        return b'\xff\xee\xdd\xcc'

    dcm = decode.DecodeCommonMetadata(decode_type=decode.DecodeType(3))
    serializer = MockSerializer()
    adt = decode.AncillaryDataTensor(dcm, serializer)
    result = adt.to_bytes()

    self.assertEqual(len(result), 20)
    self.assertEqual(result[:16], dcm.to_bytes())
    self.assertEqual(result[16:], b'\xff\xee\xdd\xcc')


if __name__ == '__main__':
  tf.test.main()
