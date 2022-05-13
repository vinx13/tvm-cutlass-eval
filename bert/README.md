First run `export.py` to dump bert model. The AutoTVM log is tuned for RTX 3070, but you can easily tune for other cards, see `autotvm.py`.

See https://github.com/apache/tvm/pull/9439 for the current benchmark numbers.


To evaluate meta schedule with auto tensorization, build TVM on [this branch](https://github.com/vinx13/tvm/tree/auto-tensorization) and run `python3 tune_bert_meta_schedule.py` 
