// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: rdma_service.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_rdma_5fservice_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_rdma_5fservice_2eproto

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
#define PROTOBUF_INTERNAL_EXPORT_rdma_5fservice_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_rdma_5fservice_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[4]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_rdma_5fservice_2eproto;
namespace ptre {
class Channel;
class ChannelDefaultTypeInternal;
extern ChannelDefaultTypeInternal _Channel_default_instance_;
class GetRemoteAddressRequest;
class GetRemoteAddressRequestDefaultTypeInternal;
extern GetRemoteAddressRequestDefaultTypeInternal _GetRemoteAddressRequest_default_instance_;
class GetRemoteAddressResponse;
class GetRemoteAddressResponseDefaultTypeInternal;
extern GetRemoteAddressResponseDefaultTypeInternal _GetRemoteAddressResponse_default_instance_;
class MemoryRegion;
class MemoryRegionDefaultTypeInternal;
extern MemoryRegionDefaultTypeInternal _MemoryRegion_default_instance_;
}  // namespace ptre
PROTOBUF_NAMESPACE_OPEN
template<> ::ptre::Channel* Arena::CreateMaybeMessage<::ptre::Channel>(Arena*);
template<> ::ptre::GetRemoteAddressRequest* Arena::CreateMaybeMessage<::ptre::GetRemoteAddressRequest>(Arena*);
template<> ::ptre::GetRemoteAddressResponse* Arena::CreateMaybeMessage<::ptre::GetRemoteAddressResponse>(Arena*);
template<> ::ptre::MemoryRegion* Arena::CreateMaybeMessage<::ptre::MemoryRegion>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace ptre {

// ===================================================================

class Channel :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.Channel) */ {
 public:
  Channel();
  virtual ~Channel();

  Channel(const Channel& from);
  Channel(Channel&& from) noexcept
    : Channel() {
    *this = ::std::move(from);
  }

  inline Channel& operator=(const Channel& from) {
    CopyFrom(from);
    return *this;
  }
  inline Channel& operator=(Channel&& from) noexcept {
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
  static const Channel& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Channel* internal_default_instance() {
    return reinterpret_cast<const Channel*>(
               &_Channel_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  void Swap(Channel* other);
  friend void swap(Channel& a, Channel& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline Channel* New() const final {
    return CreateMaybeMessage<Channel>(nullptr);
  }

  Channel* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Channel>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Channel& from);
  void MergeFrom(const Channel& from);
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
  void InternalSwap(Channel* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.Channel";
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
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_rdma_5fservice_2eproto);
    return ::descriptor_table_rdma_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // int32 lid = 1;
  void clear_lid();
  static const int kLidFieldNumber = 1;
  ::PROTOBUF_NAMESPACE_ID::int32 lid() const;
  void set_lid(::PROTOBUF_NAMESPACE_ID::int32 value);

  // int32 qpn = 2;
  void clear_qpn();
  static const int kQpnFieldNumber = 2;
  ::PROTOBUF_NAMESPACE_ID::int32 qpn() const;
  void set_qpn(::PROTOBUF_NAMESPACE_ID::int32 value);

  // uint64 snp = 4;
  void clear_snp();
  static const int kSnpFieldNumber = 4;
  ::PROTOBUF_NAMESPACE_ID::uint64 snp() const;
  void set_snp(::PROTOBUF_NAMESPACE_ID::uint64 value);

  // uint64 iid = 5;
  void clear_iid();
  static const int kIidFieldNumber = 5;
  ::PROTOBUF_NAMESPACE_ID::uint64 iid() const;
  void set_iid(::PROTOBUF_NAMESPACE_ID::uint64 value);

  // int32 psn = 3;
  void clear_psn();
  static const int kPsnFieldNumber = 3;
  ::PROTOBUF_NAMESPACE_ID::int32 psn() const;
  void set_psn(::PROTOBUF_NAMESPACE_ID::int32 value);

  // @@protoc_insertion_point(class_scope:ptre.Channel)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::int32 lid_;
  ::PROTOBUF_NAMESPACE_ID::int32 qpn_;
  ::PROTOBUF_NAMESPACE_ID::uint64 snp_;
  ::PROTOBUF_NAMESPACE_ID::uint64 iid_;
  ::PROTOBUF_NAMESPACE_ID::int32 psn_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_rdma_5fservice_2eproto;
};
// -------------------------------------------------------------------

class MemoryRegion :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.MemoryRegion) */ {
 public:
  MemoryRegion();
  virtual ~MemoryRegion();

  MemoryRegion(const MemoryRegion& from);
  MemoryRegion(MemoryRegion&& from) noexcept
    : MemoryRegion() {
    *this = ::std::move(from);
  }

  inline MemoryRegion& operator=(const MemoryRegion& from) {
    CopyFrom(from);
    return *this;
  }
  inline MemoryRegion& operator=(MemoryRegion&& from) noexcept {
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
  static const MemoryRegion& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const MemoryRegion* internal_default_instance() {
    return reinterpret_cast<const MemoryRegion*>(
               &_MemoryRegion_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  void Swap(MemoryRegion* other);
  friend void swap(MemoryRegion& a, MemoryRegion& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline MemoryRegion* New() const final {
    return CreateMaybeMessage<MemoryRegion>(nullptr);
  }

  MemoryRegion* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<MemoryRegion>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const MemoryRegion& from);
  void MergeFrom(const MemoryRegion& from);
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
  void InternalSwap(MemoryRegion* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.MemoryRegion";
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
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_rdma_5fservice_2eproto);
    return ::descriptor_table_rdma_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // uint64 remote_addr = 1;
  void clear_remote_addr();
  static const int kRemoteAddrFieldNumber = 1;
  ::PROTOBUF_NAMESPACE_ID::uint64 remote_addr() const;
  void set_remote_addr(::PROTOBUF_NAMESPACE_ID::uint64 value);

  // uint32 rkey = 2;
  void clear_rkey();
  static const int kRkeyFieldNumber = 2;
  ::PROTOBUF_NAMESPACE_ID::uint32 rkey() const;
  void set_rkey(::PROTOBUF_NAMESPACE_ID::uint32 value);

  // @@protoc_insertion_point(class_scope:ptre.MemoryRegion)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::uint64 remote_addr_;
  ::PROTOBUF_NAMESPACE_ID::uint32 rkey_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_rdma_5fservice_2eproto;
};
// -------------------------------------------------------------------

class GetRemoteAddressRequest :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.GetRemoteAddressRequest) */ {
 public:
  GetRemoteAddressRequest();
  virtual ~GetRemoteAddressRequest();

  GetRemoteAddressRequest(const GetRemoteAddressRequest& from);
  GetRemoteAddressRequest(GetRemoteAddressRequest&& from) noexcept
    : GetRemoteAddressRequest() {
    *this = ::std::move(from);
  }

  inline GetRemoteAddressRequest& operator=(const GetRemoteAddressRequest& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetRemoteAddressRequest& operator=(GetRemoteAddressRequest&& from) noexcept {
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
  static const GetRemoteAddressRequest& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GetRemoteAddressRequest* internal_default_instance() {
    return reinterpret_cast<const GetRemoteAddressRequest*>(
               &_GetRemoteAddressRequest_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  void Swap(GetRemoteAddressRequest* other);
  friend void swap(GetRemoteAddressRequest& a, GetRemoteAddressRequest& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline GetRemoteAddressRequest* New() const final {
    return CreateMaybeMessage<GetRemoteAddressRequest>(nullptr);
  }

  GetRemoteAddressRequest* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GetRemoteAddressRequest>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GetRemoteAddressRequest& from);
  void MergeFrom(const GetRemoteAddressRequest& from);
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
  void InternalSwap(GetRemoteAddressRequest* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.GetRemoteAddressRequest";
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
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_rdma_5fservice_2eproto);
    return ::descriptor_table_rdma_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .ptre.MemoryRegion mr = 3;
  int mr_size() const;
  void clear_mr();
  static const int kMrFieldNumber = 3;
  ::ptre::MemoryRegion* mutable_mr(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >*
      mutable_mr();
  const ::ptre::MemoryRegion& mr(int index) const;
  ::ptre::MemoryRegion* add_mr();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >&
      mr() const;

  // string host_name = 1;
  void clear_host_name();
  static const int kHostNameFieldNumber = 1;
  const std::string& host_name() const;
  void set_host_name(const std::string& value);
  void set_host_name(std::string&& value);
  void set_host_name(const char* value);
  void set_host_name(const char* value, size_t size);
  std::string* mutable_host_name();
  std::string* release_host_name();
  void set_allocated_host_name(std::string* host_name);

  // .ptre.Channel channel = 2;
  bool has_channel() const;
  void clear_channel();
  static const int kChannelFieldNumber = 2;
  const ::ptre::Channel& channel() const;
  ::ptre::Channel* release_channel();
  ::ptre::Channel* mutable_channel();
  void set_allocated_channel(::ptre::Channel* channel);

  // @@protoc_insertion_point(class_scope:ptre.GetRemoteAddressRequest)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion > mr_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr host_name_;
  ::ptre::Channel* channel_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_rdma_5fservice_2eproto;
};
// -------------------------------------------------------------------

class GetRemoteAddressResponse :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:ptre.GetRemoteAddressResponse) */ {
 public:
  GetRemoteAddressResponse();
  virtual ~GetRemoteAddressResponse();

  GetRemoteAddressResponse(const GetRemoteAddressResponse& from);
  GetRemoteAddressResponse(GetRemoteAddressResponse&& from) noexcept
    : GetRemoteAddressResponse() {
    *this = ::std::move(from);
  }

  inline GetRemoteAddressResponse& operator=(const GetRemoteAddressResponse& from) {
    CopyFrom(from);
    return *this;
  }
  inline GetRemoteAddressResponse& operator=(GetRemoteAddressResponse&& from) noexcept {
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
  static const GetRemoteAddressResponse& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GetRemoteAddressResponse* internal_default_instance() {
    return reinterpret_cast<const GetRemoteAddressResponse*>(
               &_GetRemoteAddressResponse_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    3;

  void Swap(GetRemoteAddressResponse* other);
  friend void swap(GetRemoteAddressResponse& a, GetRemoteAddressResponse& b) {
    a.Swap(&b);
  }

  // implements Message ----------------------------------------------

  inline GetRemoteAddressResponse* New() const final {
    return CreateMaybeMessage<GetRemoteAddressResponse>(nullptr);
  }

  GetRemoteAddressResponse* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GetRemoteAddressResponse>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GetRemoteAddressResponse& from);
  void MergeFrom(const GetRemoteAddressResponse& from);
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
  void InternalSwap(GetRemoteAddressResponse* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "ptre.GetRemoteAddressResponse";
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
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_rdma_5fservice_2eproto);
    return ::descriptor_table_rdma_5fservice_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // repeated .ptre.MemoryRegion mr = 3;
  int mr_size() const;
  void clear_mr();
  static const int kMrFieldNumber = 3;
  ::ptre::MemoryRegion* mutable_mr(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >*
      mutable_mr();
  const ::ptre::MemoryRegion& mr(int index) const;
  ::ptre::MemoryRegion* add_mr();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >&
      mr() const;

  // string host_name = 1;
  void clear_host_name();
  static const int kHostNameFieldNumber = 1;
  const std::string& host_name() const;
  void set_host_name(const std::string& value);
  void set_host_name(std::string&& value);
  void set_host_name(const char* value);
  void set_host_name(const char* value, size_t size);
  std::string* mutable_host_name();
  std::string* release_host_name();
  void set_allocated_host_name(std::string* host_name);

  // .ptre.Channel channel = 2;
  bool has_channel() const;
  void clear_channel();
  static const int kChannelFieldNumber = 2;
  const ::ptre::Channel& channel() const;
  ::ptre::Channel* release_channel();
  ::ptre::Channel* mutable_channel();
  void set_allocated_channel(::ptre::Channel* channel);

  // @@protoc_insertion_point(class_scope:ptre.GetRemoteAddressResponse)
 private:
  class HasBitSetters;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion > mr_;
  ::PROTOBUF_NAMESPACE_ID::internal::ArenaStringPtr host_name_;
  ::ptre::Channel* channel_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_rdma_5fservice_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// Channel

// int32 lid = 1;
inline void Channel::clear_lid() {
  lid_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Channel::lid() const {
  // @@protoc_insertion_point(field_get:ptre.Channel.lid)
  return lid_;
}
inline void Channel::set_lid(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  lid_ = value;
  // @@protoc_insertion_point(field_set:ptre.Channel.lid)
}

// int32 qpn = 2;
inline void Channel::clear_qpn() {
  qpn_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Channel::qpn() const {
  // @@protoc_insertion_point(field_get:ptre.Channel.qpn)
  return qpn_;
}
inline void Channel::set_qpn(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  qpn_ = value;
  // @@protoc_insertion_point(field_set:ptre.Channel.qpn)
}

// int32 psn = 3;
inline void Channel::clear_psn() {
  psn_ = 0;
}
inline ::PROTOBUF_NAMESPACE_ID::int32 Channel::psn() const {
  // @@protoc_insertion_point(field_get:ptre.Channel.psn)
  return psn_;
}
inline void Channel::set_psn(::PROTOBUF_NAMESPACE_ID::int32 value) {
  
  psn_ = value;
  // @@protoc_insertion_point(field_set:ptre.Channel.psn)
}

// uint64 snp = 4;
inline void Channel::clear_snp() {
  snp_ = PROTOBUF_ULONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 Channel::snp() const {
  // @@protoc_insertion_point(field_get:ptre.Channel.snp)
  return snp_;
}
inline void Channel::set_snp(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  
  snp_ = value;
  // @@protoc_insertion_point(field_set:ptre.Channel.snp)
}

// uint64 iid = 5;
inline void Channel::clear_iid() {
  iid_ = PROTOBUF_ULONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 Channel::iid() const {
  // @@protoc_insertion_point(field_get:ptre.Channel.iid)
  return iid_;
}
inline void Channel::set_iid(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  
  iid_ = value;
  // @@protoc_insertion_point(field_set:ptre.Channel.iid)
}

// -------------------------------------------------------------------

// MemoryRegion

// uint64 remote_addr = 1;
inline void MemoryRegion::clear_remote_addr() {
  remote_addr_ = PROTOBUF_ULONGLONG(0);
}
inline ::PROTOBUF_NAMESPACE_ID::uint64 MemoryRegion::remote_addr() const {
  // @@protoc_insertion_point(field_get:ptre.MemoryRegion.remote_addr)
  return remote_addr_;
}
inline void MemoryRegion::set_remote_addr(::PROTOBUF_NAMESPACE_ID::uint64 value) {
  
  remote_addr_ = value;
  // @@protoc_insertion_point(field_set:ptre.MemoryRegion.remote_addr)
}

// uint32 rkey = 2;
inline void MemoryRegion::clear_rkey() {
  rkey_ = 0u;
}
inline ::PROTOBUF_NAMESPACE_ID::uint32 MemoryRegion::rkey() const {
  // @@protoc_insertion_point(field_get:ptre.MemoryRegion.rkey)
  return rkey_;
}
inline void MemoryRegion::set_rkey(::PROTOBUF_NAMESPACE_ID::uint32 value) {
  
  rkey_ = value;
  // @@protoc_insertion_point(field_set:ptre.MemoryRegion.rkey)
}

// -------------------------------------------------------------------

// GetRemoteAddressRequest

// string host_name = 1;
inline void GetRemoteAddressRequest::clear_host_name() {
  host_name_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& GetRemoteAddressRequest::host_name() const {
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressRequest.host_name)
  return host_name_.GetNoArena();
}
inline void GetRemoteAddressRequest::set_host_name(const std::string& value) {
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ptre.GetRemoteAddressRequest.host_name)
}
inline void GetRemoteAddressRequest::set_host_name(std::string&& value) {
  
  host_name_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ptre.GetRemoteAddressRequest.host_name)
}
inline void GetRemoteAddressRequest::set_host_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ptre.GetRemoteAddressRequest.host_name)
}
inline void GetRemoteAddressRequest::set_host_name(const char* value, size_t size) {
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ptre.GetRemoteAddressRequest.host_name)
}
inline std::string* GetRemoteAddressRequest::mutable_host_name() {
  
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressRequest.host_name)
  return host_name_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* GetRemoteAddressRequest::release_host_name() {
  // @@protoc_insertion_point(field_release:ptre.GetRemoteAddressRequest.host_name)
  
  return host_name_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void GetRemoteAddressRequest::set_allocated_host_name(std::string* host_name) {
  if (host_name != nullptr) {
    
  } else {
    
  }
  host_name_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), host_name);
  // @@protoc_insertion_point(field_set_allocated:ptre.GetRemoteAddressRequest.host_name)
}

// .ptre.Channel channel = 2;
inline bool GetRemoteAddressRequest::has_channel() const {
  return this != internal_default_instance() && channel_ != nullptr;
}
inline void GetRemoteAddressRequest::clear_channel() {
  if (GetArenaNoVirtual() == nullptr && channel_ != nullptr) {
    delete channel_;
  }
  channel_ = nullptr;
}
inline const ::ptre::Channel& GetRemoteAddressRequest::channel() const {
  const ::ptre::Channel* p = channel_;
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressRequest.channel)
  return p != nullptr ? *p : *reinterpret_cast<const ::ptre::Channel*>(
      &::ptre::_Channel_default_instance_);
}
inline ::ptre::Channel* GetRemoteAddressRequest::release_channel() {
  // @@protoc_insertion_point(field_release:ptre.GetRemoteAddressRequest.channel)
  
  ::ptre::Channel* temp = channel_;
  channel_ = nullptr;
  return temp;
}
inline ::ptre::Channel* GetRemoteAddressRequest::mutable_channel() {
  
  if (channel_ == nullptr) {
    auto* p = CreateMaybeMessage<::ptre::Channel>(GetArenaNoVirtual());
    channel_ = p;
  }
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressRequest.channel)
  return channel_;
}
inline void GetRemoteAddressRequest::set_allocated_channel(::ptre::Channel* channel) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete channel_;
  }
  if (channel) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      channel = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, channel, submessage_arena);
    }
    
  } else {
    
  }
  channel_ = channel;
  // @@protoc_insertion_point(field_set_allocated:ptre.GetRemoteAddressRequest.channel)
}

// repeated .ptre.MemoryRegion mr = 3;
inline int GetRemoteAddressRequest::mr_size() const {
  return mr_.size();
}
inline void GetRemoteAddressRequest::clear_mr() {
  mr_.Clear();
}
inline ::ptre::MemoryRegion* GetRemoteAddressRequest::mutable_mr(int index) {
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressRequest.mr)
  return mr_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >*
GetRemoteAddressRequest::mutable_mr() {
  // @@protoc_insertion_point(field_mutable_list:ptre.GetRemoteAddressRequest.mr)
  return &mr_;
}
inline const ::ptre::MemoryRegion& GetRemoteAddressRequest::mr(int index) const {
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressRequest.mr)
  return mr_.Get(index);
}
inline ::ptre::MemoryRegion* GetRemoteAddressRequest::add_mr() {
  // @@protoc_insertion_point(field_add:ptre.GetRemoteAddressRequest.mr)
  return mr_.Add();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >&
GetRemoteAddressRequest::mr() const {
  // @@protoc_insertion_point(field_list:ptre.GetRemoteAddressRequest.mr)
  return mr_;
}

// -------------------------------------------------------------------

// GetRemoteAddressResponse

// string host_name = 1;
inline void GetRemoteAddressResponse::clear_host_name() {
  host_name_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline const std::string& GetRemoteAddressResponse::host_name() const {
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressResponse.host_name)
  return host_name_.GetNoArena();
}
inline void GetRemoteAddressResponse::set_host_name(const std::string& value) {
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), value);
  // @@protoc_insertion_point(field_set:ptre.GetRemoteAddressResponse.host_name)
}
inline void GetRemoteAddressResponse::set_host_name(std::string&& value) {
  
  host_name_.SetNoArena(
    &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::move(value));
  // @@protoc_insertion_point(field_set_rvalue:ptre.GetRemoteAddressResponse.host_name)
}
inline void GetRemoteAddressResponse::set_host_name(const char* value) {
  GOOGLE_DCHECK(value != nullptr);
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), ::std::string(value));
  // @@protoc_insertion_point(field_set_char:ptre.GetRemoteAddressResponse.host_name)
}
inline void GetRemoteAddressResponse::set_host_name(const char* value, size_t size) {
  
  host_name_.SetNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
      ::std::string(reinterpret_cast<const char*>(value), size));
  // @@protoc_insertion_point(field_set_pointer:ptre.GetRemoteAddressResponse.host_name)
}
inline std::string* GetRemoteAddressResponse::mutable_host_name() {
  
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressResponse.host_name)
  return host_name_.MutableNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline std::string* GetRemoteAddressResponse::release_host_name() {
  // @@protoc_insertion_point(field_release:ptre.GetRemoteAddressResponse.host_name)
  
  return host_name_.ReleaseNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}
inline void GetRemoteAddressResponse::set_allocated_host_name(std::string* host_name) {
  if (host_name != nullptr) {
    
  } else {
    
  }
  host_name_.SetAllocatedNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), host_name);
  // @@protoc_insertion_point(field_set_allocated:ptre.GetRemoteAddressResponse.host_name)
}

// .ptre.Channel channel = 2;
inline bool GetRemoteAddressResponse::has_channel() const {
  return this != internal_default_instance() && channel_ != nullptr;
}
inline void GetRemoteAddressResponse::clear_channel() {
  if (GetArenaNoVirtual() == nullptr && channel_ != nullptr) {
    delete channel_;
  }
  channel_ = nullptr;
}
inline const ::ptre::Channel& GetRemoteAddressResponse::channel() const {
  const ::ptre::Channel* p = channel_;
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressResponse.channel)
  return p != nullptr ? *p : *reinterpret_cast<const ::ptre::Channel*>(
      &::ptre::_Channel_default_instance_);
}
inline ::ptre::Channel* GetRemoteAddressResponse::release_channel() {
  // @@protoc_insertion_point(field_release:ptre.GetRemoteAddressResponse.channel)
  
  ::ptre::Channel* temp = channel_;
  channel_ = nullptr;
  return temp;
}
inline ::ptre::Channel* GetRemoteAddressResponse::mutable_channel() {
  
  if (channel_ == nullptr) {
    auto* p = CreateMaybeMessage<::ptre::Channel>(GetArenaNoVirtual());
    channel_ = p;
  }
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressResponse.channel)
  return channel_;
}
inline void GetRemoteAddressResponse::set_allocated_channel(::ptre::Channel* channel) {
  ::PROTOBUF_NAMESPACE_ID::Arena* message_arena = GetArenaNoVirtual();
  if (message_arena == nullptr) {
    delete channel_;
  }
  if (channel) {
    ::PROTOBUF_NAMESPACE_ID::Arena* submessage_arena = nullptr;
    if (message_arena != submessage_arena) {
      channel = ::PROTOBUF_NAMESPACE_ID::internal::GetOwnedMessage(
          message_arena, channel, submessage_arena);
    }
    
  } else {
    
  }
  channel_ = channel;
  // @@protoc_insertion_point(field_set_allocated:ptre.GetRemoteAddressResponse.channel)
}

// repeated .ptre.MemoryRegion mr = 3;
inline int GetRemoteAddressResponse::mr_size() const {
  return mr_.size();
}
inline void GetRemoteAddressResponse::clear_mr() {
  mr_.Clear();
}
inline ::ptre::MemoryRegion* GetRemoteAddressResponse::mutable_mr(int index) {
  // @@protoc_insertion_point(field_mutable:ptre.GetRemoteAddressResponse.mr)
  return mr_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >*
GetRemoteAddressResponse::mutable_mr() {
  // @@protoc_insertion_point(field_mutable_list:ptre.GetRemoteAddressResponse.mr)
  return &mr_;
}
inline const ::ptre::MemoryRegion& GetRemoteAddressResponse::mr(int index) const {
  // @@protoc_insertion_point(field_get:ptre.GetRemoteAddressResponse.mr)
  return mr_.Get(index);
}
inline ::ptre::MemoryRegion* GetRemoteAddressResponse::add_mr() {
  // @@protoc_insertion_point(field_add:ptre.GetRemoteAddressResponse.mr)
  return mr_.Add();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::ptre::MemoryRegion >&
GetRemoteAddressResponse::mr() const {
  // @@protoc_insertion_point(field_list:ptre.GetRemoteAddressResponse.mr)
  return mr_;
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace ptre

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_rdma_5fservice_2eproto
