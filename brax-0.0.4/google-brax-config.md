# google brax config

## dt and substeps

If you have bodies with very, very small mass, you need to adjust dt or/and substeps to make it
even visualize. In one env where some bodies got a mass of 0.01 i used 

```
dt: 0.015
substeps: 24
```

to visualize them after training.