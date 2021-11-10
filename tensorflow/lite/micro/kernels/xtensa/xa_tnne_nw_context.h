#ifndef __XA_TNNE_NW_CONTEXT_H__
#define __XA_TNNE_NW_CONTEXT_H__

typedef struct xa_tnne_nw_context_s {
  int init_done;
  int net_id;
  void *blob_ptr;
  void *instance_id;

} xa_tnne_nw_context;

struct XaTnneNetworkContext : public TfLiteExternalContext {
 xa_tnne_nw_context nw_ctx;
};

#endif /* __XA_TNNE_NW_CONTEXT_H__ */
