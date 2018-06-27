function y = Softmax(x)

  ex = exp(x - max(x));
  y  = ex / sum(ex);
end