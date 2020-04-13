#ifndef PTRE_LIB_CACHE_CTL_H_
#define PTRE_LIB_CACHE_CTL_H_

#define CACHE_LINE_SIZE 64 

namespace ptre {
namespace cache_ctl {

inline void mfence() {
  asm volatile("mfence":::"memory");
}

inline void clflush(char *data, int len) {
  volatile char *ptr = (char *) ((unsigned long) data &~(CACHE_LINE_SIZE - 1));
  mfence();
  for (; ptr < data + len; ptr += CACHE_LINE_SIZE) {
    //unsigned long etsc = read_tsc() + (unsigned long)(write_latency_in_ns*CPU_FREQ_MHZ/1000);
    asm volatile("clflush %0" : "+m" (*(volatile char *) ptr));
    //while (read_tsc() < etsc)
    //  cpu_pause();
  }
  mfence();
}

}  // namespace cache_ctl
}  // namespace ptre

#endif  // PTRE_LIB_CACHE_CTL_H_
