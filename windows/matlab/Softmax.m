function y = Softmax(x)

  max_val = max(x)
  
  ex = exp(x - max_val);
  y  = ex / sum(ex);
end