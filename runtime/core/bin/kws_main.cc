// Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <iostream>
#include <fstream>
#include <string>

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"

int main(int argc, char* argv[]) {
  if (argc != 6) {
    LOG(FATAL) << "Usage: kws_main fbank_dim(int) batch_size(int) threshold "
               << "kws_model_path test_wav_path";
  }

  const int num_bins = std::stoi(argv[1]);  // Fbank feature dim
  const int batch_size = std::stoi(argv[2]);
  const float threshold = std::stof(argv[3]);
  const std::string model_path = argv[4];
  const std::string wav_path = argv[5];

  const int window_shift = 0;
  int skip_size = 0;
  int frame_count = 0;

  wenet::WavReader wav_reader(wav_path);
  int num_samples = wav_reader.num_samples();
  wenet::FeaturePipelineConfig feature_config(num_bins, 16000);
  wenet::FeaturePipeline feature_pipeline(feature_config);
  std::vector<float> wav(wav_reader.data(), wav_reader.data() + num_samples);
  feature_pipeline.AcceptWaveform(wav);
  feature_pipeline.set_input_finished();

  wekws::KeywordSpotting spotter(model_path);

  std::ofstream featfile;
  featfile.open("feats_cpp.txt");

  // Simulate streaming, detect batch by batch
  while (true) {
    std::vector<std::vector<float>> feats;
    bool ok = feature_pipeline.Read(batch_size, &feats);
    std::vector<std::vector<float>> prob;

    for (int i = 0; i < feats.size(); i++) {
      for (int j = 0; j < feats[i].size(); j++) {
        if (j % 4 == 0) {
          featfile << std::endl;
        }
        featfile << feats[i][j] << " ";
      }
    }

    spotter.Forward(feats, &prob);

    for (int t = 0; t < prob.size(); t++, frame_count++) {
      if (skip_size == 0) {
        std::cout << "frame " << frame_count << " prob";
        for (int i = 0; i < prob[t].size(); i++) {
          std::cout << " " << prob[t][i];

          if (prob[t][i] >= threshold) {
            // std::cout << "keyword_index=" << i << " detected" << ", frame_count=" << frame_count << std::endl;
            // skip_size = window_shift;
            // break;
          }
        }
        std::cout << std::endl;
      } else {
        skip_size--;
        continue;
      }
    }
    // Reach the end of feature pipeline
    if (!ok) break;
  }

  featfile.close();

  return 0;
}
