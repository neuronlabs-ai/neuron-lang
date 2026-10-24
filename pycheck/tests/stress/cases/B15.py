# id: B15
# kind: leak
# expect: T001
# accept: 
# allow: 
# xfail: 
# desc: polars lead

import polars as pl

q = lf.with_columns(pl.col('close').shift(-1).alias('y'))
