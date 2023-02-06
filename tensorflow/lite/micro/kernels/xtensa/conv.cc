namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data =
      context->AllocatePersistentBuffer(context, sizeof(XtensaConvOpData));
#if defined(VISION_P6)
  if (InitXtensaContext()) {
    return nullptr;
  }
#endif  // defined(VISION_P6)

  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, ConvPrepare(context, node));

#if defined(HIFI4) || defined(HIFI5)
  TF_LITE_ENSURE_OK(context, ConvPrepareHifi(context, node));
#endif
#if defined(VISION_P6)
  TF_LITE_ENSURE_OK(context, ConvPrepareVision(context, node));
#endif  // VISION_P6
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);

  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  const auto& op_data = *(reinterpret_cast<XtensaConvOpData*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;

  TfLiteEvalTensor filter_int8;

  if (filter->type == kTfLiteInt4) {
    filter_int8.data.data = static_cast<int8_t*>(context->GetScratchBuffer(
        context, op_data.reference_op_data.filter_buffer_index));

    filter_int8.dims = filter->dims;
    filter_int8.type = kTfLiteInt8;
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(filter).FlatSize(),
        tflite::micro::GetTensorData<int8_t>(&filter_int8));

  } else {
    filter_int8 = *filter;
  }

  switch (input->type) {
    case kTfLiteInt8: {
      switch (filter_int8.type) {
        case kTfLiteInt8: {
#if defined(HIFI4) || defined(HIFI5)
          ConvEvalHifi(context, node, params, op_data, input, &filter_int8,
                       bias, output);
#elif defined(VISION_P6)
          return ConvEvalVision(context, node, params, op_data, input,
                                &filter_int8, bias, output);
#else
          reference_integer_ops::ConvPerChannel(
              ConvParamsQuantized(params, op_data.reference_op_data),
              op_data.reference_op_data.per_channel_output_multiplier,
              op_data.reference_op_data.per_channel_output_shift,
              tflite::micro::GetTensorShape(input),
              tflite::micro::GetTensorData<int8_t>(input),
              tflite::micro::GetTensorShape(filter),
              tflite::micro::GetTensorData<int8_t>(&filter_int8),
              tflite::micro::GetTensorShape(bias),
              tflite::micro::GetOptionalTensorData<int32_t>(bias),
              tflite::micro::GetTensorShape(output),
              tflite::micro::GetTensorData<int8_t>(output));
          return kTfLiteOk;
#endif
          break;
        }

        default:
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
      }
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
#if defined(HIFI4)
      ConvEvalHifi16(context, node, params, op_data, input, filter, bias,
                     output);
#else
      return ConvReferenceEvalInt16(context, node);
#endif  // defined(HIFI4)
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
