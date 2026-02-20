#include "mmap_logger.hpp"

namespace mmap_logger {

void MMapLogger::open() {
  if (opened_) return;

  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT, 0666);
  if (fd_ < 0) throw std::runtime_error("mmap_logger: open() failed: " + path_);

  map_size_ = sizeof(MMapHeader) + static_cast<size_t>(k_Cap) * sizeof(Slot);

  if (::ftruncate(fd_, static_cast<off_t>(map_size_)) != 0) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("mmap_logger: ftruncate() failed");
  }

  void* p = ::mmap(nullptr, map_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (p == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("mmap_logger: mmap() failed");
  }

  base_ = static_cast<uint8_t*>(p);
  header_ = reinterpret_cast<MMapHeader*>(base_);
  slots_ = reinterpret_cast<Slot*>(base_ + sizeof(MMapHeader));

  if (reset_) {
    std::memset(base_, 0, map_size_);

    const char kMagic[8] = {'S','T','R','L','O','G','3','\0'};
    std::memcpy(header_->magic, kMagic, 8);
    header_->version = 3;
    header_->header_size = static_cast<uint32_t>(sizeof(MMapHeader));
    header_->capacity = k_Cap;
    header_->slot_size = static_cast<uint32_t>(sizeof(Slot));
    header_->start_time_ns = 0;

    // write_count must be stored last
    atomic_store_u64(&header_->write_count, 0);
  } else {
    // Validate basic compatibility; if mismatch, reset.
    if (std::strncmp(header_->magic, "STRLOG3", 7) != 0 ||
        header_->version != 3 ||
        header_->capacity != k_Cap ||
        header_->slot_size != sizeof(Slot) ||
        header_->header_size != sizeof(MMapHeader)) {
      std::memset(base_, 0, map_size_);
      const char kMagic[8] = {'S','T','R','L','O','G','3','\0'};
      std::memcpy(header_->magic, kMagic, 8);
      header_->version = 3;
      header_->header_size = static_cast<uint32_t>(sizeof(MMapHeader));
      header_->capacity = k_Cap;
      header_->slot_size = static_cast<uint32_t>(sizeof(Slot));
      header_->start_time_ns = 0;
      atomic_store_u64(&header_->write_count, 0);
    }
  }

  opened_ = true;
}

void MMapLogger::close() {
  if (!opened_) return;

  ::msync(base_, map_size_, MS_ASYNC);
  ::munmap(base_, map_size_);
  base_ = nullptr;
  header_ = nullptr;
  slots_ = nullptr;
  map_size_ = 0;

  ::close(fd_);
  fd_ = -1;

  opened_ = false;
}

void MMapLogger::push(const LogData& x) {
  if (!opened_) open();

  const uint64_t wc = atomic_load_u64(&header_->write_count);
  const uint32_t idx = static_cast<uint32_t>(wc % k_Cap);
  Slot* s = &slots_[idx];

  // Seqlock write: seq odd -> memcpy -> seq even
  const uint64_t seq0 = atomic_load_u64(&s->seq);
  atomic_store_u64(&s->seq, seq0 + 1); // mark writing (odd)

  std::memcpy(&s->data, &x, sizeof(LogData));

  atomic_store_u64(&s->seq, seq0 + 2); // done (even)
  atomic_store_u64(&header_->write_count, wc + 1);
}

uint64_t MMapLogger::write_count() const {
  if (!opened_) return 0;
  return atomic_load_u64(&header_->write_count);
}

} // namespace mmap_logger