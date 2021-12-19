# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for official.nlp.projects.bigbird.encoder."""

import numpy as np
import tensorflow as tf

from official.projects.longformer.longformer_encoder import LongformerEncoder


class BigBirdEncoderTest(tf.test.TestCase):

  def test_encoder(self):
    sequence_length = 128
    batch_size = 2
    vocab_size = 1024
    hidden_size=256
    network = LongformerEncoder(
        vocab_size=vocab_size,
        attention_window=32,
        hidden_size=hidden_size,
        num_layers=1,
        num_attention_heads=4,
        max_sequence_length=512)
    word_id_data = np.random.randint(
        vocab_size, size=(batch_size, sequence_length))
    mask_data = np.random.randint(2, size=(batch_size, sequence_length))
    type_id_data = np.random.randint(2, size=(batch_size, sequence_length))
    inputs = {
        'input_word_ids': word_id_data,
        'input_mask': mask_data,
        'input_type_ids': type_id_data,
    }
    outputs = network(inputs)
    self.assertEqual(outputs["sequence_output"].shape,
                     (batch_size, sequence_length, hidden_size))

  # def test_save_restore(self):
  #   sequence_length = 128
  #   batch_size = 2
  #   vocab_size = 1024
  #   hidden_size = 256
  #   network = LongformerEncoder(
  #       vocab_size=vocab_size,
  #       attention_window=32,
  #       hidden_size=hidden_size,
  #       num_layers=1,
  #       num_attention_heads=4,
  #       max_sequence_length=512)
  #   word_id_data = np.random.randint(
  #       vocab_size, size=(batch_size, sequence_length))
  #   mask_data = np.random.randint(2, size=(batch_size, sequence_length))
  #   type_id_data = np.random.randint(2, size=(batch_size, sequence_length))
  #   inputs = {
  #       'input_word_ids': word_id_data,
  #       'input_mask': mask_data,
  #       'input_type_ids': type_id_data,
  #   }
  #   ref_outputs = network(inputs)
  #   model_path = self.get_temp_dir() + "/model"
  #   network.save(model_path)
  #   loaded = tf.keras.models.load_model(model_path)
  #   outputs = loaded(inputs)
  #   self.assertAllClose(outputs["sequence_output"],
  #                       ref_outputs["sequence_output"])


if __name__ == "__main__":
  tf.test.main()