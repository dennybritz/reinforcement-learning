## Implementation of A#C (Asynchronous Advantage Actor Critic)

#### Running

```
./train.py \
  --model_dir /tmp/a3c \
  --env Breakout-v0 \
  --t_max 5 \
  --eval_every 300 \
  --parallelism 8
```

See `./train.py --help` for a full list of options.