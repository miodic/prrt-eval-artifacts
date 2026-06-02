# How to eval
change `pub(crate) const NC_LIMIT: usize = 5;` at the top of incremental_search and incremental_search_full

then run 

´´´bash
cargo run --release >> results
´´´

edit main to change what you want to measure.

This produced `results_nc_limits.csv`