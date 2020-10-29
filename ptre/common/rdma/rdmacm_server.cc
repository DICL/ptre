namespace ptre {
namespace common {

RdmacmServer::RdmacmServer(const char* addr, int port) : shutdown_(false) {
  bzero(&sin_, sizeof(struct sockaddr_in));
  sin_.sin_family = AF_INET;
  sin_.sin_port = htons(port);
  if (addr) {
    inet_pton(AF_INET, addr, &sin_.sin_addr);
  } else {
    sin_.sin_addr.s_addr = INADDR_ANY;
  }
}

int RdmacmServer::Start() {
  cm_channel_ = rdma_create_event_channel();
  rdma_create_id(cm_channel_, &listen_id_, NULL, RDMA_PS_TCP);
  rdma_bind_addr(listen_id_, (struct sockaddr*) &sin_);
  rdma_listen(listen_id_, 1);

  cm_event_thread_ = std::thread([this] {
      CmEventThreadLoop();
    });
}

void RdmacmServer::CmEventThreadLoop() {
  struct rdma_cm_event* event;
  while (!shutdown_) {
    rdma_get_cm_event(cm_channel_, &event);
    switch (event->event) {
      case RDMA_CM_EVENT_CONNECT_REQUEST: {
        HandleConnectRequest(event);
        break;
      }
      case RDMA_CM_EVENT_ESTABLISHED: {
        break;
      }
      default: {
        //std::cout << "event=" << rdma_event_str(event->event) << std::endl;
        break;
      }
    }
    rdma_ack_cm_event(event);
  }
}

}  // namespace common
}  // namespace ptre
