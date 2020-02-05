REGISTER_OP("GetGlobalConsensus")
  .Input("var: resource")
  .Attr("T: numbertype")
  .Attr("var_name: string")
  .Output("gcon: resource")

REGISTER_OP("GetGlobalConsensus")
    .Input("var: Ref(T)")
    .Output("out: Ref(T)")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(GetGlobalConsensusShapeFn);

REGISTER_OP("ResourceGetGlobalConsensus")
    .Input("var: resource")
    .Attr("T: numbertype")
    .Attr("use_locking: bool = false")
    .SetShapeFn(GetGlobalConsensusShapeFn);
