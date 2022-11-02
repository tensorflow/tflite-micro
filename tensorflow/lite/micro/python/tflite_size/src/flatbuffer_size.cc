/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* This file is adopted from
 * flatbuffers/include/flatbuffers/minireflect.h **/

#include "flatbuffer_size.h"

#include <string>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/util.h"

namespace {

using flatbuffers::ElementaryType;
using flatbuffers::TypeTable;

using flatbuffers::EscapeString;
using flatbuffers::soffset_t;
using flatbuffers::String;
using flatbuffers::Table;
using flatbuffers::uoffset_t;
using flatbuffers::Vector;

using flatbuffers::FieldIndexToOffset;
using flatbuffers::GetRoot;
using flatbuffers::NumToString;
using flatbuffers::ReadScalar;

using flatbuffers::ET_BOOL;
using flatbuffers::ET_CHAR;
using flatbuffers::ET_DOUBLE;
using flatbuffers::ET_FLOAT;
using flatbuffers::ET_INT;
using flatbuffers::ET_LONG;
using flatbuffers::ET_SEQUENCE;
using flatbuffers::ET_SHORT;
using flatbuffers::ET_STRING;
using flatbuffers::ET_UCHAR;
using flatbuffers::ET_UINT;
using flatbuffers::ET_ULONG;
using flatbuffers::ET_USHORT;
using flatbuffers::ET_UTYPE;

using flatbuffers::ST_ENUM;
using flatbuffers::ST_STRUCT;
using flatbuffers::ST_TABLE;
using flatbuffers::ST_UNION;
using flatbuffers::voffset_t;

/* Utilities that petty print a tflite buffer in a json format with the
additional size information. Each element is represented by the following json
string:

field_name: {value: xxxx, total_size: }
field_name: {value: [ ], total_size: }

where value can be:
1. a dict (a new structure) that is not just value and total_size
2. a list (a new array)
3. a scalar (neither dict nor list).
*/

// Returns the storage size of a basic element type
inline size_t InlineSize(ElementaryType type, const TypeTable* type_table) {
  switch (type) {
    case ET_UTYPE:
    case ET_BOOL:
    case ET_CHAR:
    case ET_UCHAR:
      return 1;
    case ET_SHORT:
    case ET_USHORT:
      return 2;
    case ET_INT:
    case ET_UINT:
    case ET_FLOAT:
    case ET_STRING:
      return 4;
    case ET_LONG:
    case ET_ULONG:
    case ET_DOUBLE:
      return 8;
    case flatbuffers::ET_SEQUENCE:
      switch (type_table->st) {
        case ST_TABLE:
        case ST_UNION:
          return 4;
        case ST_STRUCT:
          return static_cast<size_t>(type_table->values[type_table->num_elems]);
        default:
          FLATBUFFERS_ASSERT(false);
          return 1;
      }
    default:
      FLATBUFFERS_ASSERT(false);
      return 1;
  }
}

// First, a generic iterator that can be used by multiple algorithms.
struct IterationVisitor {
  // These mark the scope of a table or struct.
  virtual void StartSequence() {}
  virtual void EndSequence(size_t) {}
  // Called for each field regardless of whether it is present or not.
  // If not present, val == nullptr. set_idx is the index of all set fields.
  virtual void Field(size_t /*field_idx*/, size_t /*set_idx*/,
                     ElementaryType /*type*/, bool /*is_vector*/,
                     const TypeTable* /*type_table*/, const char* /*name*/,
                     const uint8_t* /*val*/) {}
  // Called for a value that is actually present, after a field, or as part
  // of a vector.
  virtual size_t UType(uint8_t, const char*) {
    return InlineSize(flatbuffers::ET_UTYPE, nullptr);
  }
  virtual size_t Bool(bool) { return InlineSize(ET_BOOL, nullptr); }
  virtual size_t Char(int8_t, const char*) {
    return InlineSize(ET_CHAR, nullptr);
  }
  virtual size_t UChar(uint8_t, const char*) {
    return InlineSize(ET_UCHAR, nullptr);
  }
  virtual size_t Short(int16_t, const char*) {
    return InlineSize(ET_SHORT, nullptr);
  }
  virtual size_t UShort(uint16_t, const char*) {
    return InlineSize(ET_USHORT, nullptr);
  }
  virtual size_t Int(int32_t, const char*) {
    return InlineSize(ET_INT, nullptr);
  }
  virtual size_t UInt(uint32_t, const char*) {
    return InlineSize(ET_UINT, nullptr);
  }
  virtual size_t Long(int64_t) { return InlineSize(ET_LONG, nullptr); }
  virtual size_t ULong(uint64_t) { return InlineSize(ET_ULONG, nullptr); }
  virtual size_t Float(float) { return InlineSize(ET_FLOAT, nullptr); }
  virtual size_t Double(double) { return InlineSize(ET_DOUBLE, nullptr); }
  virtual size_t String(const String*) {
    return InlineSize(ET_STRING, nullptr);
  }
  virtual size_t Unknown(const uint8_t*) {
    return 1;
  }  // From a future version.
  // These mark the scope of a vector.
  virtual void StartVector() {}
  virtual void EndVector(size_t vector_size) {}
  virtual void Element(size_t /*i*/, ElementaryType /*type*/,
                       const TypeTable* /*type_table*/,
                       const uint8_t* /*val*/) {}
  virtual ~IterationVisitor() {}
};

inline int64_t LookupEnum(int64_t enum_val, const int64_t* values,
                          size_t num_values) {
  if (!values) return enum_val;
  for (size_t i = 0; i < num_values; i++) {
    if (enum_val == values[i]) return static_cast<int64_t>(i);
  }
  return -1;  // Unknown enum value.
}

template <typename T>
const char* EnumName(T tval, const TypeTable* type_table) {
  if (!type_table || !type_table->names) return nullptr;
  auto i = LookupEnum(static_cast<int64_t>(tval), type_table->values,
                      type_table->num_elems);
  if (i >= 0 && i < static_cast<int64_t>(type_table->num_elems)) {
    return type_table->names[i];
  }
  return nullptr;
}

size_t IterateObject(const uint8_t* obj, const TypeTable* type_table,
                     IterationVisitor* visitor);

inline size_t SizeIterateValue(ElementaryType type, const uint8_t* val,
                               const TypeTable* type_table,
                               const uint8_t* prev_val, soffset_t vector_index,
                               IterationVisitor* visitor) {
  size_t value_size = 0;
  switch (type) {
    case ET_UTYPE: {
      auto tval = ReadScalar<uint8_t>(val);
      value_size += visitor->UType(tval, EnumName(tval, type_table));
      break;
    }
    case ET_BOOL: {
      value_size += visitor->Bool(ReadScalar<uint8_t>(val) != 0);
      break;
    }
    case ET_CHAR: {
      auto tval = ReadScalar<int8_t>(val);
      value_size += visitor->Char(tval, EnumName(tval, type_table));
      break;
    }
    case ET_UCHAR: {
      auto tval = ReadScalar<uint8_t>(val);
      value_size += visitor->UChar(tval, EnumName(tval, type_table));
      break;
    }
    case ET_SHORT: {
      auto tval = ReadScalar<int16_t>(val);
      value_size += visitor->Short(tval, EnumName(tval, type_table));
      break;
    }
    case ET_USHORT: {
      auto tval = ReadScalar<uint16_t>(val);
      value_size += visitor->UShort(tval, EnumName(tval, type_table));
      break;
    }
    case ET_INT: {
      auto tval = ReadScalar<int32_t>(val);
      value_size += visitor->Int(tval, EnumName(tval, type_table));
      break;
    }
    case ET_UINT: {
      auto tval = ReadScalar<uint32_t>(val);
      value_size += visitor->UInt(tval, EnumName(tval, type_table));
      break;
    }
    case ET_LONG: {
      value_size += visitor->Long(ReadScalar<int64_t>(val));
      break;
    }
    case ET_ULONG: {
      value_size += visitor->ULong(ReadScalar<uint64_t>(val));
      break;
    }
    case ET_FLOAT: {
      value_size += visitor->Float(ReadScalar<float>(val));
      break;
    }
    case ET_DOUBLE: {
      value_size += visitor->Double(ReadScalar<double>(val));
      break;
    }
    case ET_STRING: {
      val += ReadScalar<uoffset_t>(val);
      value_size += visitor->String(reinterpret_cast<const String*>(val));
      break;
    }
    case ET_SEQUENCE: {
      switch (type_table->st) {
        case ST_TABLE:
          val += ReadScalar<uoffset_t>(val);
          value_size += IterateObject(val, type_table, visitor);
          break;
        case ST_STRUCT:
          value_size += IterateObject(val, type_table, visitor);
          break;
        case ST_UNION: {
          val += ReadScalar<uoffset_t>(val);
          FLATBUFFERS_ASSERT(prev_val);
          auto union_type = *prev_val;  // Always a uint8_t.
          if (vector_index >= 0) {
            auto type_vec = reinterpret_cast<const Vector<uint8_t>*>(prev_val);
            union_type = type_vec->Get(static_cast<uoffset_t>(vector_index));
          }
          auto type_code_idx =
              LookupEnum(union_type, type_table->values, type_table->num_elems);
          if (type_code_idx >= 0 &&
              type_code_idx < static_cast<int32_t>(type_table->num_elems)) {
            auto type_code = type_table->type_codes[type_code_idx];
            switch (type_code.base_type) {
              case ET_SEQUENCE: {
                auto ref = type_table->type_refs[type_code.sequence_ref]();
                value_size += IterateObject(val, ref, visitor);
                break;
              }
              case ET_STRING:
                value_size +=
                    visitor->String(reinterpret_cast<const String*>(val));
                break;
              default:
                value_size += visitor->Unknown(val);
            }
          } else {
            value_size += visitor->Unknown(val);
          }
          break;
        }
        case ST_ENUM:
          FLATBUFFERS_ASSERT(false);
          break;
      }
      break;
    }
    default: {
      value_size += visitor->Unknown(val);
      break;
    }
  }
  return value_size;
}

inline size_t IterateObject(const uint8_t* obj, const TypeTable* type_table,
                            IterationVisitor* visitor) {
  visitor->StartSequence();
  const uint8_t* prev_val = nullptr;
  size_t set_idx = 0;
  size_t object_size = 0;
  for (size_t i = 0; i < type_table->num_elems; i++) {
    auto type_code = type_table->type_codes[i];
    auto type = static_cast<ElementaryType>(type_code.base_type);
    auto is_vector = type_code.is_repeating != 0;
    auto ref_idx = type_code.sequence_ref;
    const TypeTable* ref = nullptr;
    if (ref_idx >= 0) {
      ref = type_table->type_refs[ref_idx]();
    }
    auto name = type_table->names ? type_table->names[i] : nullptr;
    const uint8_t* val = nullptr;
    if (type_table->st == ST_TABLE) {
      val = reinterpret_cast<const Table*>(obj)->GetAddressOf(
          FieldIndexToOffset(static_cast<voffset_t>(i)));
    } else {
      val = obj + type_table->values[i];
    }
    visitor->Field(i, set_idx, type, is_vector, ref, name, val);
    if (val) {
      set_idx++;
      if (is_vector) {
        val += ReadScalar<uoffset_t>(val);
        auto vec = reinterpret_cast<const Vector<uint8_t>*>(val);
        size_t vector_size = 0;
        visitor->StartVector();
        auto elem_ptr = vec->Data();
        for (size_t j = 0; j < vec->size(); j++) {
          visitor->Element(j, type, ref, elem_ptr);
          size_t element_size =
              SizeIterateValue(type, elem_ptr, ref, prev_val,
                               static_cast<soffset_t>(j), visitor);
          object_size += element_size;
          vector_size += element_size;
          elem_ptr += InlineSize(type, ref);
        }
        visitor->EndVector(vector_size);
      } else {
        object_size += SizeIterateValue(type, val, ref, prev_val, -1, visitor);
      }
    }
    prev_val = val;
  }
  visitor->EndSequence(object_size);
  return object_size;
}

inline void IterateFlatBuffer(const uint8_t* buffer,
                              const TypeTable* type_table,
                              IterationVisitor* callback) {
  IterateObject(GetRoot<uint8_t>(buffer), type_table, callback);
}

// Outputting a Flatbuffer to a string. Tries to conform as close to JSON /
// the output generated by idl_gen_text.cpp.

struct ToJsonWithSizeInfoVisitor : public IterationVisitor {
  std::string s;
  std::string d;   // delimiter
  bool q;          // quote
  std::string in;  // indent
  size_t indent_level;
  bool vector_delimited;
  ToJsonWithSizeInfoVisitor()
      : d("\n"), q(true), in(""), indent_level(0), vector_delimited(false) {}

  void append_indent() {
    for (size_t i = 0; i < indent_level; i++) {
      s += in;
    }
  }

  void StartSequence() override {
    s += "{ \"value\": {";
    s += d;
    indent_level++;
  }
  void EndSequence(size_t object_size) override {
    s += d;
    indent_level--;
    append_indent();

    s += "}, \"total_size\": ";
    s += NumToString(object_size);
    s += "}";
  }
  void Field(size_t /*field_idx*/, size_t set_idx, ElementaryType /*type*/,
             bool /*is_vector*/, const TypeTable* /*type_table*/,
             const char* name, const uint8_t* val) override {
    if (!val) return;
    if (set_idx) {
      s += ",";
      s += d;
    }
    append_indent();
    if (name) {
      if (q) s += "\"";
      s += name;
      if (q) s += "\"";
      s += ": ";
    }
  }
  template <typename T>
  void Named(T x, const char* name) {
    s += "{ \"value\": ";
    if (name) {
      if (q) s += "\"";
      s += name;
      if (q) s += "\"";
    } else {
      s += NumToString(x);
    }
    s += ", ";
  }

  void PrintFieldSize(uint32_t size, const char* name) {
    s += "\"total_size\":";
    s += NumToString(size);
    s += "}";
  }

  size_t UType(uint8_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_UTYPE, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t Bool(bool x) override {
    s += "{ \"value\": ";
    s += x ? "true" : "false";
    s += ", ";

    size_t size = InlineSize(ET_BOOL, nullptr);
    PrintFieldSize(size, nullptr);
    return size;
  }
  size_t Char(int8_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_UTYPE, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t UChar(uint8_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_UCHAR, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t Short(int16_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_SHORT, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t UShort(uint16_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_USHORT, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t Int(int32_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_INT, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t UInt(uint32_t x, const char* name) override {
    Named(x, name);
    size_t size = InlineSize(ET_UINT, nullptr);
    PrintFieldSize(size, name);
    return size;
  }
  size_t Long(int64_t x) override {
    Named(x, nullptr);
    size_t size = InlineSize(ET_LONG, nullptr);
    PrintFieldSize(size, nullptr);
    return size;
  }
  size_t ULong(uint64_t x) override {
    Named(x, nullptr);
    size_t size = InlineSize(ET_ULONG, nullptr);
    PrintFieldSize(size, nullptr);
    return size;
  }
  size_t Float(float x) override {
    Named(x, nullptr);
    size_t size = InlineSize(ET_FLOAT, nullptr);
    PrintFieldSize(size, nullptr);
    return size;
  }
  size_t Double(double x) override {
    Named(x, nullptr);
    size_t size = InlineSize(ET_DOUBLE, nullptr);
    PrintFieldSize(size, nullptr);

    return size;
  }
  size_t String(const struct String* str) override {
    s += "{ \"value\": ";

    EscapeString(str->c_str(), str->size(), &s, true, false);

    s += ", ";

    PrintFieldSize(str->size(), nullptr);

    // s += "}";
    return str->size();
  }
  size_t Unknown(const uint8_t*) override {
    s += "(?)";
    PrintFieldSize(1, nullptr);
    return 1;
  }
  void StartVector() override {
    s += "{ \"value\": [";
    if (vector_delimited) {
      s += d;
      indent_level++;
      append_indent();
    } else {
      s += " ";
    }
  }
  void EndVector(size_t vector_size) override {
    if (vector_delimited) {
      s += d;
      indent_level--;
      append_indent();
    } else {
      s += " ";
    }
    s += "]";
    s += ", \"total_size\": ";
    s += NumToString(vector_size);
    s += "}";
  }
  void Element(size_t i, ElementaryType /*type*/,
               const TypeTable* /*type_table*/,
               const uint8_t* /*val*/) override {
    if (i) {
      s += ",";
      if (vector_delimited) {
        s += d;
        append_indent();
      } else {
        s += " ";
      }
    }
  }
};

}  // namespace

namespace tflite {
std::string FlatBufferSizeToJsonString(
    const uint8_t* buffer, const flatbuffers::TypeTable* type_table) {
  ToJsonWithSizeInfoVisitor tostring_visitor;
  IterateFlatBuffer(buffer, type_table, &tostring_visitor);
  return tostring_visitor.s;
}

}  // namespace tflite
