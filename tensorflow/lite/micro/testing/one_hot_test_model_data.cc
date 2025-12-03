// one_hot_test_model_data.cc 같은 별도 파일로 두면 좋음

#include <cstdint>

extern "C" {

// 그냥 더미 바이트들 (유효한 TFLite 모델이 아님)
const unsigned char g_one_hot_basic_float_model[] = {
    // FlatBuffer signature 자리에는 보통 'T','F','L','3' 가 오지만
    // 여기서는 진짜 모델을 만들지 않았으니 그냥 대충 채워둔 상태입니다.
    0x54, 0x46, 0x4C, 0x33,  // 'T','F','L','3' 비슷하게 맞춰줌
    0x00, 0x00, 0x00, 0x00,  // 나머지는 전부 0
    0x00, 0x00, 0x00, 0x00,
};

const int g_one_hot_basic_float_model_len =
    sizeof(g_one_hot_basic_float_model) /
    sizeof(g_one_hot_basic_float_model[0]);

}  // extern "C"
