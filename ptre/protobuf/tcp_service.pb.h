// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tcp_service.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_tcp_5fservice_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_tcp_5fservice_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3008000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3008000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_tcp_5fservice_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_tcp_5fservice_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[2]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tcp_5fservice_2eproto;
namespace ptre {
class PushTensorRequest;
class PushTensorRequestDefaultTypeInternal;
extern PushTensorRequestDefaultTypeInternal _PushTensorRequest_default_instance_;
class PushTensorResponse;
class PushTensorResponseDefaultTypeInternal;
extern PushTensorResponseDefaultTypeInternal _PushTensorResponse_default_instance_;
}  // namespace ptre
PROTOBUF_NAMESPACE_OPEN
template<> ::ptre::PushTensorRequest* Arena::CreateMaybeMessage<::ptre::PushTensorRequest>(Arena*);
template<> ::ptre::PushTensorResponse* Arena::CreateMaybeMessage<::ptre::PushTensorResponse>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace ptre {

// ===================================================================

class PushTensorRequest :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.PushTensorRequest) */ {
 public:
  PushTensorRequest();
  virtual ~PushTensorRequest();

  PushTensorRequest(const PushTensorRequest& from);
  PushTensorRequest(PushTensorRequest&& from) noexcept
    : PushTensorRequest() {
    *this = ::std::move(from);
  }

  inline PushTensorRequest& operator=(const PushTensorRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline PushTensorRequest& operator=(PushTensorRequest&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const PushTensorRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PushTensorRequest* internal_default_instance() {
    return reinterpret_cast<const PushTensorRequest*>(
               &_PushTensorRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(PushTensorRequest* other);
  friend void swap(PushTensorRequest& a, PushTensorRequest& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline PushTensorRequest* New() const final {
    return CreateMaybeMessage<PushTensorRequest>(nullptr);
  }

  PushTensorRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<PushTensorRequest>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const PushTensorRequest& from);
  void MergeFrom(const PushTensorRequest& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(PushTensorRequest* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.PushTensorRequest";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tcp_5fservice_2eproto);
    return ::descriptor_table_tcp_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string name = 2;
  void clear_name();
  static const int kNameFieldNumber = 2;
  const std::string& name() const;
  void set_name(const std::string& value);
  void set_name(std::string&& value);
  void set_name(const char* value);
  void set_name(const char* value, size_t size);
  std::string* mutable_name();
  std::string* release_name();
  void set_allocated_name(std::string* name);

  // bytes buf = 3;
  void clear_buf();
  static const int kBufFieldNumber = 3;
  const std::string& buf() const;
  void set_buf(const std::string& value);
  void set_buf(std::string&& value);
  void set_buf(const char* value);
  void set_buf(const void* value, size_t size);
  std::string* mutable_buf();
  std::string* release_buf();
  void set_allocated_buf(std::string* buf);

  // int32 src_rank = 1;
  void clear_src_rank();
  static const int kSrcRankFieldNumber = 1;
  ::PROTOBUF_NAMESPACE_ID::int32 src_rank() const;
  void set_src_rank(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:ptre.PushTensorRequest)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr name_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr buf_;
  ::PROTOBUF_NAMESPACE_ID::int32 src_rank_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tcp_5fservice_2eproto;
};
// -------------------------------------------------------------------

class PushTensorResponse :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.PushTensorResponse) */ {
 public:
  PushTensorResponse();
  virtual ~PushTensorResponse();

  PushTensorResponse(const PushTensorResponse& from);
  PushTensorResponse(PushTensorResponse&& from) noexcept
    : PushTensorResponse() {
    *this = ::std::move(from);
  }

  inline PushTensorResponse& operator=(const PushTensorResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline PushTensorResponse& operator=(PushTensorResponse&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const PushTensorResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const PushTensorResponse* internal_default_instance() {
    return reinterpret_cast<const PushTensorResponse*>(
               &_PushTensorResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(PushTensorResponse* other);
  friend void swap(PushTensorResponse& a, PushTensorResponse& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline PushTensorResponse* New() const final {
    return CreateMaybeMessage<PushTensorResponse>(nullptr);
  }

  PushTensorResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<PushTensorResponse>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const PushTensorResponse& from);
  void MergeFrom(const PushTensorResponse& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  #if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  #else
  bool MergePartialFromCodedStream(
      ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) final;
  #endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
  void SerializeWithCachedSizes(
      ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(PushTensorResponse* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.PushTensorResponse";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_tcp_5fservice_2eproto);
    return ::descriptor_table_tcp_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // string tensor_name = 2;
  void clear_tensor_name();
  static const int kTensorNameFieldNumber = 2;
  const std::string& tensor_name() const;
  void set_tensor_name(const std::string& value);
  void set_tensor_name(std::string&& value);
  void set_tensor_name(const char* value);
  void set_tensor_name(const char* value, size_t size);
  std::string* mutable_tensor_name();
  std::string* release_tensor_name();
  void set_allocated_tensor_name(std::string* tensor_name);

  // int32 dst_rank = 1;
  void clear_dst_rank();
  static const int kDstRankFieldNumber = 1;
  ::PROTOBUF_NAMESPACE_ID::int32 dst_rank() const;
  void set_dst_rank(::PROTOBUF_NAMESPACE_ID::int32 value);

  // int32 status = 3;
  void clear_status();
  static const int kStatusFieldNumber = 3;
  ::PROTOBUF_NAMESPACE_ID::int32 status() const;
  void set_status(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:ptre.PushTensorResponse)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr tensor_name_;
  ::PROTOBUF_NAMESPACE_ID::int32 dst_rank_;
  ::PROTOBUF_NAMESPACE_ID::int32 status_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_tcp_5fservice_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// PushTensorRequest

// int32 src_rank = 1;
inline void PushTensorRequest::clear_src_rank() {
  src_rank_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 PushTensorRequest::src_rank() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorRequest.src_rank)
  return src_rank_;
}
inline void PushTensorRequest::set_src_rank(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  src_rank_ = value;
  // @@protoc_insertion_point(field_set:ptre.PushTensorRequest.src_rank)
}

// string name = 2;
inline void PushTensorRequest::clear_name() {
  name_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& PushTensorRequest::name() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorRequest.name)
  return name_.GetNoArena();
}
inline void PushTensorRequest::set_name(const std::string& value) {
  
  name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ptre.PushTensorRequest.name)
}
inline void PushTensorRequest::set_name(std::string&& value) {
  
  name_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ptre.PushTensorRequest.name)
}
inline void PushTensorRequest::set_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ptre.PushTensorRequest.name)
}
inline void PushTensorRequest::set_name(const char* value, size_t size) {
  
  name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ptre.PushTensorRequest.name)
}
inline std::string* PushTensorRequest::mutable_name() {
  
  // @@protoc_insertion_point(field_mutable:ptre.PushTensorRequest.name)
  return name_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* PushTensorRequest::release_name() {
  // @@protoc_insertion_point(field_release:ptre.PushTensorRequest.name)
  
  return name_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void PushTensorRequest::set_allocated_name(std::string* name) {
  if (name != nullptr) {
    
  } else {
    
  }
  name_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), name);
  // @@protoc_insertion_point(field_set_allocated:ptre.PushTensorRequest.name)
}

// bytes buf = 3;
inline void PushTensorRequest::clear_buf() {
  buf_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& PushTensorRequest::buf() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorRequest.buf)
  return buf_.GetNoArena();
}
inline void PushTensorRequest::set_buf(const std::string& value) {
  
  buf_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ptre.PushTensorRequest.buf)
}
inline void PushTensorRequest::set_buf(std::string&& value) {
  
  buf_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ptre.PushTensorRequest.buf)
}
inline void PushTensorRequest::set_buf(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  buf_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ptre.PushTensorRequest.buf)
}
inline void PushTensorRequest::set_buf(const void* value, size_t size) {
  
  buf_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ptre.PushTensorRequest.buf)
}
inline std::string* PushTensorRequest::mutable_buf() {
  
  // @@protoc_insertion_point(field_mutable:ptre.PushTensorRequest.buf)
  return buf_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* PushTensorRequest::release_buf() {
  // @@protoc_insertion_point(field_release:ptre.PushTensorRequest.buf)
  
  return buf_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void PushTensorRequest::set_allocated_buf(std::string* buf) {
  if (buf != nullptr) {
    
  } else {
    
  }
  buf_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), buf);
  // @@protoc_insertion_point(field_set_allocated:ptre.PushTensorRequest.buf)
}

// -------------------------------------------------------------------

// PushTensorResponse

// int32 dst_rank = 1;
inline void PushTensorResponse::clear_dst_rank() {
  dst_rank_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 PushTensorResponse::dst_rank() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorResponse.dst_rank)
  return dst_rank_;
}
inline void PushTensorResponse::set_dst_rank(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  dst_rank_ = value;
  // @@protoc_insertion_point(field_set:ptre.PushTensorResponse.dst_rank)
}

// string tensor_name = 2;
inline void PushTensorResponse::clear_tensor_name() {
  tensor_name_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& PushTensorResponse::tensor_name() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorResponse.tensor_name)
  return tensor_name_.GetNoArena();
}
inline void PushTensorResponse::set_tensor_name(const std::string& value) {
  
  tensor_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ptre.PushTensorResponse.tensor_name)
}
inline void PushTensorResponse::set_tensor_name(std::string&& value) {
  
  tensor_name_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ptre.PushTensorResponse.tensor_name)
}
inline void PushTensorResponse::set_tensor_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  tensor_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ptre.PushTensorResponse.tensor_name)
}
inline void PushTensorResponse::set_tensor_name(const char* value, size_t size) {
  
  tensor_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ptre.PushTensorResponse.tensor_name)
}
inline std::string* PushTensorResponse::mutable_tensor_name() {
  
  // @@protoc_insertion_point(field_mutable:ptre.PushTensorResponse.tensor_name)
  return tensor_name_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* PushTensorResponse::release_tensor_name() {
  // @@protoc_insertion_point(field_release:ptre.PushTensorResponse.tensor_name)
  
  return tensor_name_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void PushTensorResponse::set_allocated_tensor_name(std::string* tensor_name) {
  if (tensor_name != nullptr) {
    
  } else {
    
  }
  tensor_name_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), tensor_name);
  // @@protoc_insertion_point(field_set_allocated:ptre.PushTensorResponse.tensor_name)
}

// int32 status = 3;
inline void PushTensorResponse::clear_status() {
  status_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 PushTensorResponse::status() const {
  // @@protoc_insertion_point(field_get:ptre.PushTensorResponse.status)
  return status_;
}
inline void PushTensorResponse::set_status(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  status_ = value;
  // @@protoc_insertion_point(field_set:ptre.PushTensorResponse.status)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace ptre

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_tcp_5fservice_2eproto