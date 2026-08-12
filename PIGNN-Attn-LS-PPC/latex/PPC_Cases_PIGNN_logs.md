# 13

- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn
    
    ```python
    (adl23_2) no12neni@cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ ls
    collate_blockdiag_optimized_complex_columns.py  Dataset_optimized_complex_columns.py  GNSMsg_armijo.py  GNSMsg_SelfAttention_armijo.py  helper.py  read_npy_columns_optimized.py  sbatch  train_valid_test.py
    (adl23_2) no12neni@cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn
    MODEL:GNSMsg_EdgeSelfAttn, PINN:True, Block:True, d:4, d_hi:16, K:40, Runname:case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.8, PARQUET:['../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet'], BATCH:16, EP:100, LR:0.0001
    Using device: cuda
    Processing file ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet ...
    Traceback (most recent call last):
      File "/home/cip/ai2023/no12neni/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/train_valid_test.py", line 142, in <module>
        full_ds = ChanghunDataset(PARQUET, per_unit=PER_UNIT, device=None)
      File "/home/cip/ai2023/no12neni/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC/Dataset_optimized_complex_columns.py", line 191, in __init__
        df = pd.read_parquet(path, engine="pyarrow")
      File "/proj/aimi-adl/envs/adl23_2/lib/python3.9/site-packages/pandas/io/parquet.py", line 503, in read_parquet
        return impl.read(
      File "/proj/aimi-adl/envs/adl23_2/lib/python3.9/site-packages/pandas/io/parquet.py", line 244, in read
        path_or_handle, handles, kwargs["filesystem"] = _get_path_or_handle(
      File "/proj/aimi-adl/envs/adl23_2/lib/python3.9/site-packages/pandas/io/parquet.py", line 102, in _get_path_or_handle
        handles = get_handle(
      File "/proj/aimi-adl/envs/adl23_2/lib/python3.9/site-packages/pandas/io/common.py", line 865, in get_handle
        handle = open(handle, ioargs.mode)
    FileNotFoundError: [Errno 2] No such file or directory: '../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet'
    (adl23_2) no12neni@cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn
    MODEL:GNSMsg_EdgeSelfAttn, PINN:True, Block:True, d:4, d_hi:16, K:40, Runname:case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333, PARQUET:['../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet'], BATCH:16, EP:100, LR:0.0001
    Using device: cuda
    Processing file ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet ...
    Decoded columns from ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet: ['bus_typ', 'vn_kv', 'Y_shunt_bus', 'Branch_f_bus', 'Branch_t_bus', 'Branch_status', 'Branch_tau', 'Branch_shift_deg', 'Branch_y_series_from', 'Branch_y_series_to', 'Branch_y_series_ft', 'Branch_y_shunt_from', 'Branch_y_shunt_to', 'Is_trafo', 'Branch_hv_is_f', 'Branch_n', 'Y_Lines', 'Y_C_Lines', 'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton']
    Raw shape: (36000, 28)
    Final combined dataset shape: (36000, 28)
    Processing file ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet ...
    Decoded columns from ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet: ['bus_typ', 'vn_kv', 'Y_shunt_bus', 'Branch_f_bus', 'Branch_t_bus', 'Branch_status', 'Branch_tau', 'Branch_shift_deg', 'Branch_y_series_from', 'Branch_y_series_to', 'Branch_y_series_ft', 'Branch_y_shunt_from', 'Branch_y_shunt_to', 'Is_trafo', 'Branch_hv_is_f', 'Branch_n', 'Y_Lines', 'Y_C_Lines', 'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton']
    Raw shape: (36000, 28)
    Final combined dataset shape: (36000, 28)
    Dataset sizes | train 11998   valid 11998   test 12004
    Total number of parameters: 7696
    Initial metrics before training:
    Epoch   0 | train loss 4.4524e+01  rmse 3.3246e-01 (mag 6.8285e-02, ang 1.8643e+01°) | valid loss 4.4939e+01  rmse 3.3240e-01 (mag 6.8357e-02, ang 1.8638e+01°)
    Epoch   1 | train loss 3.9018e-01  rmse 7.3945e-02 (mag 7.1701e-03, ang 4.2167e+00°) | valid loss 3.1539e-02  rmse 3.4135e-03 (mag 6.4097e-04, ang 1.9210e-01°) | time 71.40s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   2 | train loss 3.0499e-02  rmse 5.0448e-03 (mag 9.8157e-04, ang 2.8352e-01°) | valid loss 3.0055e-02  rmse 3.7330e-03 (mag 7.8359e-04, ang 2.0912e-01°) | time 70.83s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   3 | train loss 2.9402e-02  rmse 4.6111e-03 (mag 2.1208e-03, ang 2.3459e-01°) | valid loss 2.9447e-02  rmse 3.6368e-03 (mag 2.8474e-03, ang 1.2962e-01°) | time 70.63s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   4 | train loss 2.9057e-02  rmse 4.2127e-03 (mag 3.0683e-03, ang 1.6539e-01°) | valid loss 2.9224e-02  rmse 4.7107e-03 (mag 3.3647e-03, ang 1.8890e-01°) | time 70.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   5 | train loss 2.8921e-02  rmse 5.2088e-03 (mag 3.1641e-03, ang 2.3707e-01°) | valid loss 2.9104e-02  rmse 3.2360e-03 (mag 2.8725e-03, ang 8.5376e-02°) | time 71.23s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   6 | train loss 2.8782e-02  rmse 4.3719e-03 (mag 2.9343e-03, ang 1.8569e-01°) | valid loss 2.9015e-02  rmse 3.2099e-03 (mag 2.8636e-03, ang 8.3096e-02°) | time 71.09s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   7 | train loss 2.8718e-02  rmse 5.3254e-03 (mag 3.2956e-03, ang 2.3967e-01°) | valid loss 2.8921e-02  rmse 4.1979e-03 (mag 3.5193e-03, ang 1.3112e-01°) | time 70.62s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   8 | train loss 2.8618e-02  rmse 4.3444e-03 (mag 3.4034e-03, ang 1.5471e-01°) | valid loss 2.8870e-02  rmse 3.5118e-03 (mag 3.1631e-03, ang 8.7416e-02°) | time 70.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   9 | train loss 2.8562e-02  rmse 4.5702e-03 (mag 3.1981e-03, ang 1.8706e-01°) | valid loss 2.8802e-02  rmse 4.3790e-03 (mag 3.1374e-03, ang 1.7503e-01°) | time 70.75s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  10 | train loss 2.8508e-02  rmse 4.6499e-03 (mag 3.0720e-03, ang 2.0000e-01°) | valid loss 2.8754e-02  rmse 4.5257e-03 (mag 3.0088e-03, ang 1.9370e-01°) | time 70.62s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  11 | train loss 2.8466e-02  rmse 4.7533e-03 (mag 2.8602e-03, ang 2.1752e-01°) | valid loss 2.8720e-02  rmse 3.8330e-03 (mag 2.5477e-03, ang 1.6408e-01°) | time 70.41s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  12 | train loss 2.8425e-02  rmse 4.7728e-03 (mag 2.6106e-03, ang 2.2893e-01°) | valid loss 2.8682e-02  rmse 5.2633e-03 (mag 2.3675e-03, ang 2.6934e-01°) | time 70.31s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  13 | train loss 2.8390e-02  rmse 4.7778e-03 (mag 2.2181e-03, ang 2.4246e-01°) | valid loss 2.8655e-02  rmse 5.8039e-03 (mag 1.9717e-03, ang 3.1276e-01°) | time 70.50s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  14 | train loss 2.8361e-02  rmse 4.7395e-03 (mag 1.8241e-03, ang 2.5064e-01°) | valid loss 2.8619e-02  rmse 4.4669e-03 (mag 1.6285e-03, ang 2.3832e-01°) | time 71.13s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  15 | train loss 2.8337e-02  rmse 4.7078e-03 (mag 1.4473e-03, ang 2.5667e-01°) | valid loss 2.8597e-02  rmse 4.5796e-03 (mag 1.2330e-03, ang 2.5270e-01°) | time 70.38s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  16 | train loss 2.8318e-02  rmse 4.6709e-03 (mag 1.1820e-03, ang 2.5891e-01°) | valid loss 2.8581e-02  rmse 4.8531e-03 (mag 1.1057e-03, ang 2.7075e-01°) | time 70.58s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  17 | train loss 2.8305e-02  rmse 4.6499e-03 (mag 9.4960e-04, ang 2.6080e-01°) | valid loss 2.8570e-02  rmse 4.4826e-03 (mag 8.0869e-04, ang 2.5262e-01°) | time 71.42s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  18 | train loss 2.8295e-02  rmse 4.6366e-03 (mag 7.9250e-04, ang 2.6175e-01°) | valid loss 2.8563e-02  rmse 4.8041e-03 (mag 7.4103e-04, ang 2.7196e-01°) | time 70.59s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  19 | train loss 2.8289e-02  rmse 4.6248e-03 (mag 6.9166e-04, ang 2.6200e-01°) | valid loss 2.8559e-02  rmse 4.6744e-03 (mag 6.5889e-04, ang 2.6515e-01°) | time 70.34s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  20 | train loss 2.8287e-02  rmse 4.6242e-03 (mag 6.5015e-04, ang 2.6232e-01°) | valid loss 2.8557e-02  rmse 4.5481e-03 (mag 6.2257e-04, ang 2.5814e-01°) | time 70.18s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  21 | train loss 2.8638e-02  rmse 8.1201e-03 (mag 1.8565e-03, ang 4.5293e-01°) | valid loss 2.8556e-02  rmse 2.7951e-03 (mag 2.1840e-03, ang 9.9939e-02°) | time 70.10s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  22 | train loss 2.8227e-02  rmse 5.2720e-03 (mag 2.0790e-03, ang 2.7758e-01°) | valid loss 2.8437e-02  rmse 4.7728e-03 (mag 2.0961e-03, ang 2.4568e-01°) | time 70.48s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  23 | train loss 2.8153e-02  rmse 4.6483e-03 (mag 2.0114e-03, ang 2.4010e-01°) | valid loss 2.8373e-02  rmse 4.5977e-03 (mag 1.8061e-03, ang 2.4225e-01°) | time 70.34s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  24 | train loss 2.8088e-02  rmse 4.1583e-03 (mag 1.7413e-03, ang 2.1636e-01°) | valid loss 2.8344e-02  rmse 5.1862e-03 (mag 1.2854e-03, ang 2.8788e-01°) | time 70.24s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  25 | train loss 2.8035e-02  rmse 3.8655e-03 (mag 1.6041e-03, ang 2.0151e-01°) | valid loss 2.8275e-02  rmse 2.4610e-03 (mag 1.8168e-03, ang 9.5117e-02°) | time 70.40s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  26 | train loss 2.7985e-02  rmse 3.6999e-03 (mag 1.6133e-03, ang 1.9078e-01°) | valid loss 2.8263e-02  rmse 3.4949e-03 (mag 1.7612e-03, ang 1.7296e-01°) | time 69.99s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  27 | train loss 2.7941e-02  rmse 3.5952e-03 (mag 1.7054e-03, ang 1.8134e-01°) | valid loss 2.8203e-02  rmse 3.9747e-03 (mag 1.6279e-03, ang 2.0776e-01°) | time 68.84s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  28 | train loss 2.7906e-02  rmse 3.5014e-03 (mag 1.6952e-03, ang 1.7554e-01°) | valid loss 2.8148e-02  rmse 2.6544e-03 (mag 1.8130e-03, ang 1.1109e-01°) | time 68.78s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  29 | train loss 2.7872e-02  rmse 3.4988e-03 (mag 1.7487e-03, ang 1.7363e-01°) | valid loss 2.8117e-02  rmse 3.5246e-03 (mag 1.8008e-03, ang 1.7360e-01°) | time 68.72s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  30 | train loss 2.7839e-02  rmse 3.5244e-03 (mag 1.8024e-03, ang 1.7353e-01°) | valid loss 2.8093e-02  rmse 3.5472e-03 (mag 1.7339e-03, ang 1.7730e-01°) | time 68.90s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  31 | train loss 2.7815e-02  rmse 3.5382e-03 (mag 1.8383e-03, ang 1.7322e-01°) | valid loss 2.8072e-02  rmse 4.3203e-03 (mag 1.9142e-03, ang 2.2191e-01°) | time 68.74s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  32 | train loss 2.7792e-02  rmse 3.5853e-03 (mag 1.8752e-03, ang 1.7508e-01°) | valid loss 2.8056e-02  rmse 4.5097e-03 (mag 1.9847e-03, ang 2.3202e-01°) | time 68.62s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  33 | train loss 2.7772e-02  rmse 3.6507e-03 (mag 1.9363e-03, ang 1.7733e-01°) | valid loss 2.8038e-02  rmse 3.3289e-03 (mag 1.8562e-03, ang 1.5833e-01°) | time 68.68s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  34 | train loss 2.7757e-02  rmse 3.7009e-03 (mag 1.9737e-03, ang 1.7937e-01°) | valid loss 2.8029e-02  rmse 4.3117e-03 (mag 1.9202e-03, ang 2.2119e-01°) | time 68.61s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  35 | train loss 2.7744e-02  rmse 3.7301e-03 (mag 1.9934e-03, ang 1.8064e-01°) | valid loss 2.8009e-02  rmse 3.6732e-03 (mag 1.9259e-03, ang 1.7921e-01°) | time 69.30s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  36 | train loss 2.7734e-02  rmse 3.7860e-03 (mag 2.0206e-03, ang 1.8344e-01°) | valid loss 2.8003e-02  rmse 3.3331e-03 (mag 1.9425e-03, ang 1.5519e-01°) | time 68.65s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  37 | train loss 2.7726e-02  rmse 3.7992e-03 (mag 2.0155e-03, ang 1.8452e-01°) | valid loss 2.7996e-02  rmse 3.7883e-03 (mag 1.9996e-03, ang 1.8435e-01°) | time 68.70s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  38 | train loss 2.7721e-02  rmse 3.8177e-03 (mag 2.0252e-03, ang 1.8543e-01°) | valid loss 2.7994e-02  rmse 3.5867e-03 (mag 2.0269e-03, ang 1.6954e-01°) | time 68.54s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  39 | train loss 2.7718e-02  rmse 3.8221e-03 (mag 2.0233e-03, ang 1.8579e-01°) | valid loss 2.7990e-02  rmse 3.7802e-03 (mag 2.0353e-03, ang 1.8252e-01°) | time 68.94s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  40 | train loss 2.7716e-02  rmse 3.8314e-03 (mag 2.0199e-03, ang 1.8654e-01°) | valid loss 2.7988e-02  rmse 3.8413e-03 (mag 2.0015e-03, ang 1.8785e-01°) | time 68.89s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  41 | train loss 2.7919e-02  rmse 5.7471e-03 (mag 1.5393e-03, ang 3.1725e-01°) | valid loss 2.7983e-02  rmse 4.4317e-03 (mag 1.3501e-03, ang 2.4185e-01°) | time 68.68s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  42 | train loss 2.7702e-02  rmse 2.9726e-03 (mag 1.2357e-03, ang 1.5490e-01°) | valid loss 2.7983e-02  rmse 3.4358e-03 (mag 6.2239e-04, ang 1.9360e-01°) | time 68.89s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  43 | train loss 2.7676e-02  rmse 3.1045e-03 (mag 7.8941e-04, ang 1.7203e-01°) | valid loss 2.7944e-02  rmse 2.9295e-03 (mag 6.4847e-04, ang 1.6369e-01°) | time 68.72s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  44 | train loss 2.7648e-02  rmse 3.1126e-03 (mag 4.1389e-04, ang 1.7675e-01°) | valid loss 2.7893e-02  rmse 3.6120e-03 (mag 2.7144e-04, ang 2.0637e-01°) | time 69.00s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  45 | train loss 2.7623e-02  rmse 3.4057e-03 (mag 5.6755e-04, ang 1.9240e-01°) | valid loss 2.7890e-02  rmse 5.1753e-03 (mag 8.1277e-04, ang 2.9284e-01°) | time 68.57s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  46 | train loss 2.7601e-02  rmse 3.5680e-03 (mag 9.0211e-04, ang 1.9779e-01°) | valid loss 2.7863e-02  rmse 5.7265e-03 (mag 9.8054e-04, ang 3.2326e-01°) | time 68.50s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  47 | train loss 2.7578e-02  rmse 3.8122e-03 (mag 1.1768e-03, ang 2.0775e-01°) | valid loss 2.7837e-02  rmse 3.2177e-03 (mag 1.3472e-03, ang 1.6742e-01°) | time 68.51s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  48 | train loss 2.7560e-02  rmse 3.8632e-03 (mag 1.3748e-03, ang 2.0686e-01°) | valid loss 2.7821e-02  rmse 2.6208e-03 (mag 1.5137e-03, ang 1.2258e-01°) | time 69.08s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  49 | train loss 2.7540e-02  rmse 3.9573e-03 (mag 1.5835e-03, ang 2.0779e-01°) | valid loss 2.7805e-02  rmse 4.3608e-03 (mag 1.6211e-03, ang 2.3195e-01°) | time 68.37s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  50 | train loss 2.7521e-02  rmse 3.9470e-03 (mag 1.7061e-03, ang 2.0393e-01°) | valid loss 2.7783e-02  rmse 3.4682e-03 (mag 1.7593e-03, ang 1.7125e-01°) | time 68.47s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  51 | train loss 2.7507e-02  rmse 3.9286e-03 (mag 1.7843e-03, ang 2.0054e-01°) | valid loss 2.7764e-02  rmse 3.6235e-03 (mag 1.8035e-03, ang 1.8007e-01°) | time 68.72s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  52 | train loss 2.7491e-02  rmse 3.8904e-03 (mag 1.8619e-03, ang 1.9572e-01°) | valid loss 2.7752e-02  rmse 3.5757e-03 (mag 1.8344e-03, ang 1.7586e-01°) | time 68.70s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  53 | train loss 2.7478e-02  rmse 3.8558e-03 (mag 1.9144e-03, ang 1.9177e-01°) | valid loss 2.7749e-02  rmse 3.8872e-03 (mag 1.9927e-03, ang 1.9123e-01°) | time 69.05s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  54 | train loss 2.7467e-02  rmse 3.8049e-03 (mag 1.9635e-03, ang 1.8674e-01°) | valid loss 2.7730e-02  rmse 3.8209e-03 (mag 2.0076e-03, ang 1.8627e-01°) | time 68.79s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  55 | train loss 2.7458e-02  rmse 3.7657e-03 (mag 2.0148e-03, ang 1.8228e-01°) | valid loss 2.7723e-02  rmse 3.8499e-03 (mag 2.0518e-03, ang 1.8664e-01°) | time 68.40s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  56 | train loss 2.7450e-02  rmse 3.7276e-03 (mag 2.0423e-03, ang 1.7866e-01°) | valid loss 2.7717e-02  rmse 3.7118e-03 (mag 2.0114e-03, ang 1.7874e-01°) | time 68.73s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  57 | train loss 2.7444e-02  rmse 3.7133e-03 (mag 2.0794e-03, ang 1.7627e-01°) | valid loss 2.7713e-02  rmse 3.9619e-03 (mag 2.1596e-03, ang 1.9031e-01°) | time 69.05s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  58 | train loss 2.7440e-02  rmse 3.6860e-03 (mag 2.1051e-03, ang 1.7336e-01°) | valid loss 2.7711e-02  rmse 3.6542e-03 (mag 2.1049e-03, ang 1.7115e-01°) | time 68.46s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  59 | train loss 2.7438e-02  rmse 3.6761e-03 (mag 2.1197e-03, ang 1.7209e-01°) | valid loss 2.7709e-02  rmse 3.7783e-03 (mag 2.1379e-03, ang 1.7849e-01°) | time 68.67s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  60 | train loss 2.7437e-02  rmse 3.6651e-03 (mag 2.1248e-03, ang 1.7110e-01°) | valid loss 2.7709e-02  rmse 3.6758e-03 (mag 2.1252e-03, ang 1.7184e-01°) | time 68.72s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  61 | train loss 2.7741e-02  rmse 8.4094e-03 (mag 4.5716e-03, ang 4.0441e-01°) | valid loss 2.7769e-02  rmse 5.0633e-03 (mag 4.6437e-03, ang 1.1562e-01°) | time 68.85s
    Epoch  62 | train loss 2.7445e-02  rmse 4.9871e-03 (mag 4.4779e-03, ang 1.2578e-01°) | valid loss 2.7706e-02  rmse 4.9435e-03 (mag 4.1189e-03, ang 1.5663e-01°) | time 68.65s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  63 | train loss 2.7423e-02  rmse 5.0912e-03 (mag 4.4952e-03, ang 1.3696e-01°) | valid loss 2.7660e-02  rmse 4.7570e-03 (mag 4.5978e-03, ang 6.9919e-02°) | time 68.48s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  64 | train loss 2.7393e-02  rmse 5.1246e-03 (mag 4.5686e-03, ang 1.3302e-01°) | valid loss 2.7650e-02  rmse 5.8621e-03 (mag 4.5988e-03, ang 2.0829e-01°) | time 68.62s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  65 | train loss 2.7377e-02  rmse 5.3527e-03 (mag 4.6301e-03, ang 1.5389e-01°) | valid loss 2.7625e-02  rmse 4.7878e-03 (mag 4.6733e-03, ang 5.9623e-02°) | time 68.67s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  66 | train loss 2.7357e-02  rmse 5.3450e-03 (mag 4.6537e-03, ang 1.5064e-01°) | valid loss 2.7610e-02  rmse 4.9214e-03 (mag 4.7261e-03, ang 7.8634e-02°) | time 69.15s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  67 | train loss 2.7337e-02  rmse 5.4795e-03 (mag 4.7198e-03, ang 1.5949e-01°) | valid loss 2.7599e-02  rmse 4.9930e-03 (mag 4.8426e-03, ang 6.9705e-02°) | time 68.66s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  68 | train loss 2.7324e-02  rmse 5.6447e-03 (mag 4.7158e-03, ang 1.7774e-01°) | valid loss 2.7575e-02  rmse 5.4246e-03 (mag 4.6466e-03, ang 1.6038e-01°) | time 68.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  69 | train loss 2.7306e-02  rmse 5.7380e-03 (mag 4.7420e-03, ang 1.8510e-01°) | valid loss 2.7568e-02  rmse 5.4348e-03 (mag 4.7091e-03, ang 1.5545e-01°) | time 69.28s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  70 | train loss 2.7293e-02  rmse 5.8824e-03 (mag 4.7333e-03, ang 2.0012e-01°) | valid loss 2.7564e-02  rmse 6.7991e-03 (mag 4.8564e-03, ang 2.7264e-01°) | time 69.37s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  71 | train loss 2.7281e-02  rmse 6.0259e-03 (mag 4.7400e-03, ang 2.1319e-01°) | valid loss 2.7577e-02  rmse 6.0933e-03 (mag 4.7173e-03, ang 2.2098e-01°) | time 68.54s
    Epoch  72 | train loss 2.7269e-02  rmse 6.1605e-03 (mag 4.7171e-03, ang 2.2703e-01°) | valid loss 2.7538e-02  rmse 6.1658e-03 (mag 4.7331e-03, ang 2.2640e-01°) | time 68.55s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  73 | train loss 2.7259e-02  rmse 6.3178e-03 (mag 4.7266e-03, ang 2.4019e-01°) | valid loss 2.7524e-02  rmse 6.1924e-03 (mag 4.7257e-03, ang 2.2928e-01°) | time 68.50s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  74 | train loss 2.7251e-02  rmse 6.4320e-03 (mag 4.7063e-03, ang 2.5120e-01°) | valid loss 2.7522e-02  rmse 6.8089e-03 (mag 4.6973e-03, ang 2.8242e-01°) | time 68.61s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  75 | train loss 2.7244e-02  rmse 6.5486e-03 (mag 4.6982e-03, ang 2.6138e-01°) | valid loss 2.7515e-02  rmse 6.7052e-03 (mag 4.6678e-03, ang 2.7580e-01°) | time 68.67s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  76 | train loss 2.7239e-02  rmse 6.6642e-03 (mag 4.6947e-03, ang 2.7100e-01°) | valid loss 2.7508e-02  rmse 6.4099e-03 (mag 4.6927e-03, ang 2.5018e-01°) | time 68.60s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  77 | train loss 2.7235e-02  rmse 6.7421e-03 (mag 4.6938e-03, ang 2.7731e-01°) | valid loss 2.7507e-02  rmse 6.8850e-03 (mag 4.7026e-03, ang 2.8813e-01°) | time 68.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  78 | train loss 2.7232e-02  rmse 6.8068e-03 (mag 4.6868e-03, ang 2.8283e-01°) | valid loss 2.7505e-02  rmse 6.8002e-03 (mag 4.6753e-03, ang 2.8293e-01°) | time 68.52s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  79 | train loss 2.7230e-02  rmse 6.8403e-03 (mag 4.6870e-03, ang 2.8546e-01°) | valid loss 2.7503e-02  rmse 6.8816e-03 (mag 4.6784e-03, ang 2.8916e-01°) | time 68.84s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  80 | train loss 2.7229e-02  rmse 6.8690e-03 (mag 4.6872e-03, ang 2.8770e-01°) | valid loss 2.7503e-02  rmse 6.9414e-03 (mag 4.6661e-03, ang 2.9445e-01°) | time 68.50s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  81 | train loss 2.7568e-02  rmse 7.8988e-03 (mag 4.2648e-03, ang 3.8093e-01°) | valid loss 2.7582e-02  rmse 9.1423e-03 (mag 4.3559e-03, ang 4.6054e-01°) | time 68.49s
    Epoch  82 | train loss 2.7259e-02  rmse 6.2160e-03 (mag 4.8562e-03, ang 2.2232e-01°) | valid loss 2.7512e-02  rmse 5.2215e-03 (mag 4.7244e-03, ang 1.2740e-01°) | time 68.39s
    Epoch  83 | train loss 2.7249e-02  rmse 6.3539e-03 (mag 4.8954e-03, ang 2.3208e-01°) | valid loss 2.7491e-02  rmse 5.8767e-03 (mag 4.8800e-03, ang 1.8761e-01°) | time 69.12s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  84 | train loss 2.7233e-02  rmse 6.1120e-03 (mag 4.9000e-03, ang 2.0932e-01°) | valid loss 2.7540e-02  rmse 8.1279e-03 (mag 4.4110e-03, ang 3.9115e-01°) | time 68.53s
    Epoch  85 | train loss 2.7205e-02  rmse 6.0197e-03 (mag 4.8277e-03, ang 2.0603e-01°) | valid loss 2.7456e-02  rmse 5.7936e-03 (mag 4.7598e-03, ang 1.8926e-01°) | time 68.53s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  86 | train loss 2.7190e-02  rmse 6.0636e-03 (mag 4.8757e-03, ang 2.0654e-01°) | valid loss 2.7460e-02  rmse 6.2420e-03 (mag 4.7616e-03, ang 2.3124e-01°) | time 68.43s
    Epoch  87 | train loss 2.7174e-02  rmse 6.1577e-03 (mag 4.8731e-03, ang 2.1568e-01°) | valid loss 2.7441e-02  rmse 5.2333e-03 (mag 4.9067e-03, ang 1.0426e-01°) | time 68.71s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  88 | train loss 2.7166e-02  rmse 6.1443e-03 (mag 4.8236e-03, ang 2.1807e-01°) | valid loss 2.7430e-02  rmse 6.6761e-03 (mag 4.8195e-03, ang 2.6470e-01°) | time 68.61s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  89 | train loss 2.7156e-02  rmse 6.2503e-03 (mag 4.7917e-03, ang 2.2994e-01°) | valid loss 2.7410e-02  rmse 5.5709e-03 (mag 4.7794e-03, ang 1.6400e-01°) | time 68.34s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  90 | train loss 2.7140e-02  rmse 6.1342e-03 (mag 4.7841e-03, ang 2.1998e-01°) | valid loss 2.7401e-02  rmse 6.0182e-03 (mag 4.8245e-03, ang 2.0612e-01°) | time 68.69s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  91 | train loss 2.7130e-02  rmse 6.1243e-03 (mag 4.7549e-03, ang 2.2115e-01°) | valid loss 2.7395e-02  rmse 5.7215e-03 (mag 4.7947e-03, ang 1.7888e-01°) | time 68.58s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  92 | train loss 2.7122e-02  rmse 6.0920e-03 (mag 4.7315e-03, ang 2.1987e-01°) | valid loss 2.7392e-02  rmse 6.1323e-03 (mag 4.6563e-03, ang 2.2864e-01°) | time 68.94s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  93 | train loss 2.7114e-02  rmse 6.0910e-03 (mag 4.7028e-03, ang 2.2179e-01°) | valid loss 2.7384e-02  rmse 5.7457e-03 (mag 4.7493e-03, ang 1.8528e-01°) | time 68.51s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  94 | train loss 2.7107e-02  rmse 6.0344e-03 (mag 4.6716e-03, ang 2.1886e-01°) | valid loss 2.7378e-02  rmse 6.2582e-03 (mag 4.6316e-03, ang 2.4114e-01°) | time 68.63s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  95 | train loss 2.7102e-02  rmse 6.0183e-03 (mag 4.6581e-03, ang 2.1834e-01°) | valid loss 2.7371e-02  rmse 6.1422e-03 (mag 4.6422e-03, ang 2.3045e-01°) | time 68.57s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  96 | train loss 2.7097e-02  rmse 5.9891e-03 (mag 4.6399e-03, ang 2.1698e-01°) | valid loss 2.7369e-02  rmse 5.9748e-03 (mag 4.6320e-03, ang 2.1623e-01°) | time 68.83s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  97 | train loss 2.7094e-02  rmse 5.9763e-03 (mag 4.6302e-03, ang 2.1649e-01°) | valid loss 2.7368e-02  rmse 5.8844e-03 (mag 4.6350e-03, ang 2.0772e-01°) | time 68.74s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  98 | train loss 2.7091e-02  rmse 5.9598e-03 (mag 4.6118e-03, ang 2.1629e-01°) | valid loss 2.7366e-02  rmse 5.8724e-03 (mag 4.5875e-03, ang 2.1005e-01°) | time 68.57s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  99 | train loss 2.7090e-02  rmse 5.9517e-03 (mag 4.6064e-03, ang 2.1594e-01°) | valid loss 2.7365e-02  rmse 5.8895e-03 (mag 4.6100e-03, ang 2.1000e-01°) | time 68.48s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch 100 | train loss 2.7089e-02  rmse 5.9538e-03 (mag 4.6061e-03, ang 2.1615e-01°) | valid loss 2.7364e-02  rmse 5.9591e-03 (mag 4.6013e-03, ang 2.1696e-01°) | time 68.75s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    
    Test physics-loss : 2.7136e-02 | total RMSE : 5.9589e-03 | |V| RMSE : 4.6018e-03 | θ RMSE : 2.1691e-01°
    (adl23_2) no12neni@cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ Connection to cip7b0.cip.cs.fau.de closed by remote host.
    Connection to cip7b0.cip.cs.fau.de closed.
    client_loop: send disconnect: Broken pipe
    (GridAssist) changhunkim@Mac sbatch % 
    
    ```
    
    Test physics-loss : 2.7136e-02 | total RMSE : 5.9589e-03 | |V| RMSE : 4.6018e-03 | θ RMSE : 2.1691e-01°
    (adl23_2) no12neni@cip7b0:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ Connection to cip7b0.cip.cs.fau.de closed by remote host.
    Connection to cip7b0.cip.cs.fau.de closed.
    client_loop: send disconnect: Broken pipe
    (GridAssist) changhunkim@Mac sbatch %
    
- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn
    
    ```python
    (adl23_2) no12neni@cip7b1:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn
    MODEL:GNSMsg_EdgeSelfAttn, PINN:True, Block:True, d:4, d_hi:16, K:40, Runname:case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333, PARQUET:['../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet'], BATCH:16, EP:100, LR:0.0001
    Using device: cuda
    Processing file ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet ...
    Decoded columns from ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet: ['bus_typ', 'vn_kv', 'Y_shunt_bus', 'Branch_f_bus', 'Branch_t_bus', 'Branch_status', 'Branch_tau', 'Branch_shift_deg', 'Branch_y_series_from', 'Branch_y_series_to', 'Branch_y_series_ft', 'Branch_y_shunt_from', 'Branch_y_shunt_to', 'Is_trafo', 'Branch_hv_is_f', 'Branch_n', 'Y_Lines', 'Y_C_Lines', 'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton']
    Raw shape: (36000, 28)
    Final combined dataset shape: (36000, 28)
    Processing file ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet ...
    Decoded columns from ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet: ['bus_typ', 'vn_kv', 'Y_shunt_bus', 'Branch_f_bus', 'Branch_t_bus', 'Branch_status', 'Branch_tau', 'Branch_shift_deg', 'Branch_y_series_from', 'Branch_y_series_to', 'Branch_y_series_ft', 'Branch_y_shunt_from', 'Branch_y_shunt_to', 'Is_trafo', 'Branch_hv_is_f', 'Branch_n', 'Y_Lines', 'Y_C_Lines', 'Y_matrix', 'u_start', 'u_newton', 'S_start', 'S_newton']
    Raw shape: (36000, 28)
    Final combined dataset shape: (36000, 28)
    Dataset sizes | train 11998   valid 11998   test 12004
    Total number of parameters: 7696
    Initial metrics before training:
    Epoch   0 | train loss 4.6203e+00  rmse 2.7413e-01 (mag 2.0738e-02, ang 1.5661e+01°) | valid loss 4.6644e+00  rmse 2.7327e-01 (mag 2.0835e-02, ang 1.5612e+01°)
    Epoch   1 | train loss 2.9714e-01  rmse 8.7362e-02 (mag 8.2187e-03, ang 4.9832e+00°) | valid loss 6.2735e-02  rmse 4.9291e-03 (mag 9.2576e-04, ang 2.7739e-01°) | time 107.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   2 | train loss 6.1035e-02  rmse 6.1097e-03 (mag 9.2002e-04, ang 3.4607e-01°) | valid loss 6.0931e-02  rmse 4.3778e-03 (mag 6.9475e-04, ang 2.4765e-01°) | time 106.96s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   3 | train loss 5.9599e-02  rmse 6.0388e-03 (mag 7.0136e-04, ang 3.4366e-01°) | valid loss 5.9600e-02  rmse 3.5381e-03 (mag 5.2792e-04, ang 2.0045e-01°) | time 108.90s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   4 | train loss 5.8946e-02  rmse 5.5137e-03 (mag 6.2494e-04, ang 3.1388e-01°) | valid loss 5.9352e-02  rmse 3.6111e-03 (mag 5.3712e-04, ang 2.0460e-01°) | time 106.33s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   5 | train loss 5.8706e-02  rmse 4.2390e-03 (mag 4.6079e-04, ang 2.4144e-01°) | valid loss 5.9214e-02  rmse 3.7774e-03 (mag 4.1926e-04, ang 2.1509e-01°) | time 110.39s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   6 | train loss 5.8571e-02  rmse 3.3369e-03 (mag 4.1383e-04, ang 1.8971e-01°) | valid loss 5.9146e-02  rmse 4.4066e-03 (mag 5.0262e-04, ang 2.5083e-01°) | time 108.51s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   7 | train loss 5.8500e-02  rmse 3.9110e-03 (mag 4.6861e-04, ang 2.2247e-01°) | valid loss 5.8988e-02  rmse 2.1386e-03 (mag 2.2293e-04, ang 1.2187e-01°) | time 106.87s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   8 | train loss 5.8374e-02  rmse 2.6507e-03 (mag 3.1467e-04, ang 1.5080e-01°) | valid loss 5.8943e-02  rmse 3.4867e-03 (mag 3.1815e-04, ang 1.9894e-01°) | time 107.10s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch   9 | train loss 5.8313e-02  rmse 3.0783e-03 (mag 3.1158e-04, ang 1.7547e-01°) | valid loss 5.8860e-02  rmse 2.5540e-03 (mag 2.9808e-04, ang 1.4534e-01°) | time 107.07s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  10 | train loss 5.8228e-02  rmse 2.5469e-03 (mag 2.7743e-04, ang 1.4506e-01°) | valid loss 5.8807e-02  rmse 4.1867e-03 (mag 3.7469e-04, ang 2.3892e-01°) | time 106.89s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  11 | train loss 5.8159e-02  rmse 2.2099e-03 (mag 2.7712e-04, ang 1.2562e-01°) | valid loss 5.8715e-02  rmse 2.4924e-03 (mag 2.7262e-04, ang 1.4195e-01°) | time 107.83s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  12 | train loss 5.8097e-02  rmse 1.9941e-03 (mag 2.2081e-04, ang 1.1355e-01°) | valid loss 5.8642e-02  rmse 1.4334e-03 (mag 1.4630e-04, ang 8.1697e-02°) | time 107.09s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  13 | train loss 5.8072e-02  rmse 2.4207e-03 (mag 2.8380e-04, ang 1.3774e-01°) | valid loss 5.8614e-02  rmse 1.9235e-03 (mag 1.9817e-04, ang 1.0962e-01°) | time 107.27s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  14 | train loss 5.8011e-02  rmse 1.8412e-03 (mag 2.0861e-04, ang 1.0481e-01°) | valid loss 5.8570e-02  rmse 1.7642e-03 (mag 1.6517e-04, ang 1.0064e-01°) | time 106.31s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  15 | train loss 5.7975e-02  rmse 1.5686e-03 (mag 1.7512e-04, ang 8.9310e-02°) | valid loss 5.8543e-02  rmse 1.7825e-03 (mag 1.9723e-04, ang 1.0150e-01°) | time 106.66s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  16 | train loss 5.7947e-02  rmse 1.4486e-03 (mag 1.5786e-04, ang 8.2502e-02°) | valid loss 5.8513e-02  rmse 1.5302e-03 (mag 1.4179e-04, ang 8.7298e-02°) | time 106.63s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  17 | train loss 5.7927e-02  rmse 1.3298e-03 (mag 1.4063e-04, ang 7.5762e-02°) | valid loss 5.8494e-02  rmse 1.1714e-03 (mag 1.1926e-04, ang 6.6769e-02°) | time 106.45s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  18 | train loss 5.7919e-02  rmse 1.3682e-03 (mag 1.4034e-04, ang 7.7977e-02°) | valid loss 5.8485e-02  rmse 1.3137e-03 (mag 1.3169e-04, ang 7.4889e-02°) | time 106.38s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  19 | train loss 5.7906e-02  rmse 1.2026e-03 (mag 1.2372e-04, ang 6.8541e-02°) | valid loss 5.8477e-02  rmse 1.0791e-03 (mag 1.1591e-04, ang 6.1472e-02°) | time 105.95s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  20 | train loss 5.7902e-02  rmse 1.1573e-03 (mag 1.2930e-04, ang 6.5896e-02°) | valid loss 5.8475e-02  rmse 1.1209e-03 (mag 1.2425e-04, ang 6.3829e-02°) | time 105.88s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  21 | train loss 5.8107e-02  rmse 5.9893e-03 (mag 8.0831e-04, ang 3.4002e-01°) | valid loss 5.8491e-02  rmse 3.2384e-03 (mag 3.2566e-04, ang 1.8461e-01°) | time 104.16s
    Epoch  22 | train loss 5.7849e-02  rmse 3.3035e-03 (mag 3.9760e-04, ang 1.8790e-01°) | valid loss 5.8334e-02  rmse 2.0665e-03 (mag 3.3981e-04, ang 1.1679e-01°) | time 104.50s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  23 | train loss 5.7708e-02  rmse 2.4124e-03 (mag 2.7297e-04, ang 1.3733e-01°) | valid loss 5.8220e-02  rmse 1.7081e-03 (mag 2.1954e-04, ang 9.7057e-02°) | time 104.47s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  24 | train loss 5.7652e-02  rmse 3.2378e-03 (mag 3.5358e-04, ang 1.8440e-01°) | valid loss 5.8233e-02  rmse 4.7115e-03 (mag 2.5582e-04, ang 2.6955e-01°) | time 104.21s
    Epoch  25 | train loss 5.7563e-02  rmse 2.5463e-03 (mag 2.7184e-04, ang 1.4506e-01°) | valid loss 5.8134e-02  rmse 2.7402e-03 (mag 2.7310e-04, ang 1.5622e-01°) | time 105.02s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  26 | train loss 5.7502e-02  rmse 2.4319e-03 (mag 2.8634e-04, ang 1.3837e-01°) | valid loss 5.8073e-02  rmse 2.6697e-03 (mag 3.3029e-04, ang 1.5179e-01°) | time 104.96s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  27 | train loss 5.7472e-02  rmse 3.0697e-03 (mag 3.2560e-04, ang 1.7489e-01°) | valid loss 5.8005e-02  rmse 2.0041e-03 (mag 2.0532e-04, ang 1.1422e-01°) | time 104.90s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  28 | train loss 5.7401e-02  rmse 2.0658e-03 (mag 2.1298e-04, ang 1.1773e-01°) | valid loss 5.7947e-02  rmse 2.5789e-03 (mag 2.0332e-04, ang 1.4730e-01°) | time 106.24s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  29 | train loss 5.7360e-02  rmse 2.0352e-03 (mag 2.2967e-04, ang 1.1587e-01°) | valid loss 5.7989e-02  rmse 4.2920e-03 (mag 3.3768e-04, ang 2.4515e-01°) | time 106.25s
    Epoch  30 | train loss 5.7324e-02  rmse 1.9404e-03 (mag 2.0507e-04, ang 1.1056e-01°) | valid loss 5.7895e-02  rmse 3.3082e-03 (mag 1.4588e-04, ang 1.8936e-01°) | time 105.30s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  31 | train loss 5.7295e-02  rmse 1.8750e-03 (mag 1.9748e-04, ang 1.0683e-01°) | valid loss 5.7846e-02  rmse 1.7336e-03 (mag 1.3057e-04, ang 9.9045e-02°) | time 105.71s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  32 | train loss 5.7262e-02  rmse 1.4401e-03 (mag 1.6158e-04, ang 8.1988e-02°) | valid loss 5.7832e-02  rmse 1.5882e-03 (mag 1.7542e-04, ang 9.0442e-02°) | time 107.00s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  33 | train loss 5.7243e-02  rmse 1.5179e-03 (mag 1.5818e-04, ang 8.6494e-02°) | valid loss 5.7813e-02  rmse 1.0365e-03 (mag 1.8237e-04, ang 5.8460e-02°) | time 106.49s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  34 | train loss 5.7219e-02  rmse 1.1480e-03 (mag 1.1908e-04, ang 6.5418e-02°) | valid loss 5.7784e-02  rmse 7.9682e-04 (mag 9.5358e-05, ang 4.5326e-02°) | time 106.07s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  35 | train loss 5.7205e-02  rmse 1.1081e-03 (mag 1.2233e-04, ang 6.3103e-02°) | valid loss 5.7776e-02  rmse 7.7611e-04 (mag 1.0304e-04, ang 4.4074e-02°) | time 107.04s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  36 | train loss 5.7191e-02  rmse 7.5338e-04 (mag 1.0186e-04, ang 4.2769e-02°) | valid loss 5.7766e-02  rmse 7.4325e-04 (mag 8.6238e-05, ang 4.2297e-02°) | time 106.95s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  37 | train loss 5.7184e-02  rmse 6.7641e-04 (mag 8.6879e-05, ang 3.8435e-02°) | valid loss 5.7758e-02  rmse 5.4382e-04 (mag 8.8036e-05, ang 3.0748e-02°) | time 107.10s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  38 | train loss 5.7177e-02  rmse 5.4990e-04 (mag 7.4545e-05, ang 3.1216e-02°) | valid loss 5.7754e-02  rmse 5.1809e-04 (mag 7.2474e-05, ang 2.9393e-02°) | time 106.95s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  39 | train loss 5.7174e-02  rmse 5.2842e-04 (mag 7.0112e-05, ang 3.0009e-02°) | valid loss 5.7752e-02  rmse 5.2692e-04 (mag 7.0916e-05, ang 2.9916e-02°) | time 105.99s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  40 | train loss 5.7173e-02  rmse 5.8658e-04 (mag 7.1311e-05, ang 3.3359e-02°) | valid loss 5.7750e-02  rmse 4.9635e-04 (mag 6.9036e-05, ang 2.8162e-02°) | time 105.86s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  41 | train loss 5.7479e-02  rmse 6.2324e-03 (mag 7.6278e-04, ang 3.5441e-01°) | valid loss 5.7796e-02  rmse 2.9660e-03 (mag 4.0628e-04, ang 1.6834e-01°) | time 104.15s
    Epoch  42 | train loss 5.7212e-02  rmse 2.9304e-03 (mag 3.3800e-04, ang 1.6678e-01°) | valid loss 5.7791e-02  rmse 1.8074e-03 (mag 3.3823e-04, ang 1.0172e-01°) | time 104.00s
    Epoch  43 | train loss 5.7182e-02  rmse 2.8290e-03 (mag 3.4489e-04, ang 1.6088e-01°) | valid loss 5.7751e-02  rmse 2.2364e-03 (mag 6.1846e-04, ang 1.2314e-01°) | time 107.06s
    Epoch  44 | train loss 5.7143e-02  rmse 2.5246e-03 (mag 2.9738e-04, ang 1.4364e-01°) | valid loss 5.7739e-02  rmse 4.7319e-03 (mag 2.3986e-04, ang 2.7077e-01°) | time 104.44s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  45 | train loss 5.7100e-02  rmse 1.9957e-03 (mag 2.2925e-04, ang 1.1359e-01°) | valid loss 5.7656e-02  rmse 1.5410e-03 (mag 2.0507e-04, ang 8.7507e-02°) | time 107.08s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  46 | train loss 5.7075e-02  rmse 1.7954e-03 (mag 2.3266e-04, ang 1.0200e-01°) | valid loss 5.7672e-02  rmse 3.2377e-03 (mag 2.3659e-04, ang 1.8501e-01°) | time 104.80s
    Epoch  47 | train loss 5.7065e-02  rmse 2.4634e-03 (mag 2.5730e-04, ang 1.4037e-01°) | valid loss 5.7645e-02  rmse 2.3961e-03 (mag 2.1519e-04, ang 1.3673e-01°) | time 103.21s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  48 | train loss 5.7062e-02  rmse 2.8549e-03 (mag 3.0854e-04, ang 1.6262e-01°) | valid loss 5.7591e-02  rmse 1.0183e-03 (mag 1.4293e-04, ang 5.7768e-02°) | time 104.85s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  49 | train loss 5.7017e-02  rmse 1.7684e-03 (mag 1.8669e-04, ang 1.0075e-01°) | valid loss 5.7594e-02  rmse 9.6734e-04 (mag 1.3782e-04, ang 5.4859e-02°) | time 102.51s
    Epoch  50 | train loss 5.6992e-02  rmse 1.1730e-03 (mag 1.3350e-04, ang 6.6772e-02°) | valid loss 5.7565e-02  rmse 7.6172e-04 (mag 1.0301e-04, ang 4.3242e-02°) | time 103.02s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  51 | train loss 5.6977e-02  rmse 1.1308e-03 (mag 1.2292e-04, ang 6.4405e-02°) | valid loss 5.7557e-02  rmse 1.7336e-03 (mag 1.8246e-04, ang 9.8778e-02°) | time 104.20s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  52 | train loss 5.6965e-02  rmse 1.0678e-03 (mag 1.1858e-04, ang 6.0800e-02°) | valid loss 5.7534e-02  rmse 1.3307e-03 (mag 1.0674e-04, ang 7.5998e-02°) | time 106.14s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  53 | train loss 5.6952e-02  rmse 9.1650e-04 (mag 1.0895e-04, ang 5.2139e-02°) | valid loss 5.7521e-02  rmse 6.3434e-04 (mag 1.2578e-04, ang 3.5623e-02°) | time 105.83s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  54 | train loss 5.6942e-02  rmse 8.6081e-04 (mag 1.0525e-04, ang 4.8950e-02°) | valid loss 5.7512e-02  rmse 5.9466e-04 (mag 7.6397e-05, ang 3.3789e-02°) | time 103.63s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  55 | train loss 5.6932e-02  rmse 6.6908e-04 (mag 7.9345e-05, ang 3.8065e-02°) | valid loss 5.7506e-02  rmse 7.8843e-04 (mag 8.1757e-05, ang 4.4930e-02°) | time 103.63s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  56 | train loss 5.6925e-02  rmse 6.4010e-04 (mag 7.4548e-05, ang 3.6425e-02°) | valid loss 5.7502e-02  rmse 4.8379e-04 (mag 6.7556e-05, ang 2.7447e-02°) | time 103.80s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  57 | train loss 5.6920e-02  rmse 5.5691e-04 (mag 6.4026e-05, ang 3.1697e-02°) | valid loss 5.7495e-02  rmse 4.2698e-04 (mag 5.7442e-05, ang 2.4242e-02°) | time 103.49s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  58 | train loss 5.6916e-02  rmse 5.3044e-04 (mag 6.0950e-05, ang 3.0191e-02°) | valid loss 5.7493e-02  rmse 3.7012e-04 (mag 5.5406e-05, ang 2.0968e-02°) | time 103.78s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  59 | train loss 5.6914e-02  rmse 4.8898e-04 (mag 5.7888e-05, ang 2.7819e-02°) | valid loss 5.7492e-02  rmse 4.1150e-04 (mag 5.6178e-05, ang 2.3357e-02°) | time 103.06s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  60 | train loss 5.6913e-02  rmse 4.6978e-04 (mag 5.7295e-05, ang 2.6715e-02°) | valid loss 5.7491e-02  rmse 4.0107e-04 (mag 5.4269e-05, ang 2.2768e-02°) | time 103.28s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  61 | train loss 5.7083e-02  rmse 4.6262e-03 (mag 6.4805e-04, ang 2.6245e-01°) | valid loss 5.7556e-02  rmse 3.3549e-03 (mag 2.9602e-04, ang 1.9147e-01°) | time 101.20s
    Epoch  62 | train loss 5.6948e-02  rmse 2.2198e-03 (mag 2.6394e-04, ang 1.2628e-01°) | valid loss 5.7509e-02  rmse 2.4345e-03 (mag 2.0923e-04, ang 1.3897e-01°) | time 100.83s
    Epoch  63 | train loss 5.6922e-02  rmse 2.1251e-03 (mag 2.5477e-04, ang 1.2088e-01°) | valid loss 5.7463e-02  rmse 1.0309e-03 (mag 1.3704e-04, ang 5.8539e-02°) | time 100.92s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  64 | train loss 5.6890e-02  rmse 1.7784e-03 (mag 2.0355e-04, ang 1.0122e-01°) | valid loss 5.7456e-02  rmse 2.3082e-03 (mag 1.9171e-04, ang 1.3179e-01°) | time 101.37s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  65 | train loss 5.6866e-02  rmse 1.6472e-03 (mag 1.6896e-04, ang 9.3882e-02°) | valid loss 5.7425e-02  rmse 2.1457e-03 (mag 1.2907e-04, ang 1.2272e-01°) | time 100.86s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  66 | train loss 5.6854e-02  rmse 2.0147e-03 (mag 2.0702e-04, ang 1.1482e-01°) | valid loss 5.7401e-02  rmse 1.3763e-03 (mag 1.5770e-04, ang 7.8335e-02°) | time 101.75s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  67 | train loss 5.6828e-02  rmse 1.3322e-03 (mag 1.4909e-04, ang 7.5853e-02°) | valid loss 5.7390e-02  rmse 1.3485e-03 (mag 1.1676e-04, ang 7.6974e-02°) | time 101.45s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  68 | train loss 5.6809e-02  rmse 1.4006e-03 (mag 1.4049e-04, ang 7.9846e-02°) | valid loss 5.7391e-02  rmse 2.9987e-03 (mag 3.6018e-04, ang 1.7057e-01°) | time 100.81s
    Epoch  69 | train loss 5.6802e-02  rmse 1.8781e-03 (mag 1.9945e-04, ang 1.0700e-01°) | valid loss 5.7362e-02  rmse 9.9668e-04 (mag 1.0554e-04, ang 5.6784e-02°) | time 100.91s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  70 | train loss 5.6774e-02  rmse 1.0187e-03 (mag 1.1462e-04, ang 5.7998e-02°) | valid loss 5.7339e-02  rmse 7.2542e-04 (mag 7.7254e-05, ang 4.1327e-02°) | time 101.56s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  71 | train loss 5.6757e-02  rmse 7.9277e-04 (mag 8.9450e-05, ang 4.5133e-02°) | valid loss 5.7331e-02  rmse 4.9247e-04 (mag 7.2398e-05, ang 2.7910e-02°) | time 102.21s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  72 | train loss 5.6747e-02  rmse 8.9030e-04 (mag 9.4823e-05, ang 5.0720e-02°) | valid loss 5.7315e-02  rmse 7.6970e-04 (mag 8.7785e-05, ang 4.3813e-02°) | time 100.90s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  73 | train loss 5.6737e-02  rmse 8.3017e-04 (mag 8.9090e-05, ang 4.7290e-02°) | valid loss 5.7312e-02  rmse 1.0536e-03 (mag 1.0878e-04, ang 6.0043e-02°) | time 101.73s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  74 | train loss 5.6727e-02  rmse 6.9767e-04 (mag 7.7332e-05, ang 3.9727e-02°) | valid loss 5.7302e-02  rmse 7.0404e-04 (mag 6.5575e-05, ang 4.0163e-02°) | time 101.79s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  75 | train loss 5.6719e-02  rmse 6.6835e-04 (mag 7.3241e-05, ang 3.8063e-02°) | valid loss 5.7294e-02  rmse 4.7353e-04 (mag 5.3950e-05, ang 2.6955e-02°) | time 101.31s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  76 | train loss 5.6713e-02  rmse 5.5125e-04 (mag 6.1395e-05, ang 3.1388e-02°) | valid loss 5.7287e-02  rmse 4.7430e-04 (mag 6.4564e-05, ang 2.6922e-02°) | time 101.76s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  77 | train loss 5.6708e-02  rmse 5.6710e-04 (mag 6.1359e-05, ang 3.2302e-02°) | valid loss 5.7285e-02  rmse 6.2812e-04 (mag 6.6761e-05, ang 3.5785e-02°) | time 101.51s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  78 | train loss 5.6705e-02  rmse 5.0330e-04 (mag 5.3470e-05, ang 2.8674e-02°) | valid loss 5.7282e-02  rmse 3.4618e-04 (mag 4.4810e-05, ang 1.9668e-02°) | time 101.32s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  79 | train loss 5.6703e-02  rmse 4.2777e-04 (mag 4.8137e-05, ang 2.4354e-02°) | valid loss 5.7279e-02  rmse 3.6538e-04 (mag 4.4131e-05, ang 2.0782e-02°) | time 101.49s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  80 | train loss 5.6702e-02  rmse 3.9912e-04 (mag 4.4409e-05, ang 2.2726e-02°) | valid loss 5.7279e-02  rmse 3.2408e-04 (mag 4.2660e-05, ang 1.8407e-02°) | time 101.34s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  81 | train loss 5.6854e-02  rmse 4.2826e-03 (mag 5.8917e-04, ang 2.4304e-01°) | valid loss 5.7342e-02  rmse 1.4305e-03 (mag 1.6396e-04, ang 8.1422e-02°) | time 100.14s
    Epoch  82 | train loss 5.6736e-02  rmse 2.1326e-03 (mag 2.4482e-04, ang 1.2138e-01°) | valid loss 5.7309e-02  rmse 3.5515e-03 (mag 2.8124e-04, ang 2.0285e-01°) | time 100.79s
    Epoch  83 | train loss 5.6706e-02  rmse 1.6392e-03 (mag 1.8488e-04, ang 9.3320e-02°) | valid loss 5.7287e-02  rmse 2.1929e-03 (mag 1.7876e-04, ang 1.2522e-01°) | time 101.10s
    Epoch  84 | train loss 5.6729e-02  rmse 3.1894e-03 (mag 3.1012e-04, ang 1.8187e-01°) | valid loss 5.7256e-02  rmse 1.8047e-03 (mag 1.4201e-04, ang 1.0308e-01°) | time 100.96s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  85 | train loss 5.6673e-02  rmse 1.4669e-03 (mag 1.4537e-04, ang 8.3634e-02°) | valid loss 5.7226e-02  rmse 8.1914e-04 (mag 9.7019e-05, ang 4.6603e-02°) | time 101.36s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  86 | train loss 5.6654e-02  rmse 1.2713e-03 (mag 1.2955e-04, ang 7.2464e-02°) | valid loss 5.7257e-02  rmse 1.3638e-03 (mag 1.0238e-04, ang 7.7921e-02°) | time 101.42s
    Epoch  87 | train loss 5.6639e-02  rmse 1.4705e-03 (mag 1.4067e-04, ang 8.3866e-02°) | valid loss 5.7192e-02  rmse 7.3995e-04 (mag 7.5783e-05, ang 4.2173e-02°) | time 101.24s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  88 | train loss 5.6621e-02  rmse 1.0706e-03 (mag 1.1632e-04, ang 6.0975e-02°) | valid loss 5.7183e-02  rmse 1.3144e-03 (mag 9.5728e-05, ang 7.5110e-02°) | time 101.42s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  89 | train loss 5.6610e-02  rmse 1.1305e-03 (mag 1.2274e-04, ang 6.4388e-02°) | valid loss 5.7171e-02  rmse 5.7345e-04 (mag 6.4523e-05, ang 3.2648e-02°) | time 102.12s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  90 | train loss 5.6595e-02  rmse 1.0261e-03 (mag 1.0257e-04, ang 5.8499e-02°) | valid loss 5.7171e-02  rmse 9.3658e-04 (mag 9.1969e-05, ang 5.3403e-02°) | time 102.03s
    Epoch  91 | train loss 5.6583e-02  rmse 1.0070e-03 (mag 9.7677e-05, ang 5.7423e-02°) | valid loss 5.7150e-02  rmse 5.6254e-04 (mag 5.9679e-05, ang 3.2049e-02°) | time 101.62s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  92 | train loss 5.6572e-02  rmse 7.4974e-04 (mag 7.9544e-05, ang 4.2714e-02°) | valid loss 5.7140e-02  rmse 5.4926e-04 (mag 7.3431e-05, ang 3.1188e-02°) | time 100.97s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  93 | train loss 5.6563e-02  rmse 6.4632e-04 (mag 7.0672e-05, ang 3.6809e-02°) | valid loss 5.7135e-02  rmse 8.8242e-04 (mag 5.7018e-05, ang 5.0453e-02°) | time 101.83s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  94 | train loss 5.6556e-02  rmse 6.2625e-04 (mag 6.7419e-05, ang 3.5673e-02°) | valid loss 5.7128e-02  rmse 5.6684e-04 (mag 5.0083e-05, ang 3.2351e-02°) | time 101.13s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  95 | train loss 5.6549e-02  rmse 5.2392e-04 (mag 5.5880e-05, ang 2.9847e-02°) | valid loss 5.7124e-02  rmse 4.4080e-04 (mag 4.6611e-05, ang 2.5115e-02°) | time 101.07s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  96 | train loss 5.6545e-02  rmse 5.1509e-04 (mag 5.5433e-05, ang 2.9341e-02°) | valid loss 5.7121e-02  rmse 6.8784e-04 (mag 5.3882e-05, ang 3.9290e-02°) | time 101.38s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  97 | train loss 5.6541e-02  rmse 4.8410e-04 (mag 5.1221e-05, ang 2.7581e-02°) | valid loss 5.7121e-02  rmse 4.6568e-04 (mag 5.3547e-05, ang 2.6505e-02°) | time 101.19s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  98 | train loss 5.6538e-02  rmse 4.4240e-04 (mag 4.8211e-05, ang 2.5197e-02°) | valid loss 5.7116e-02  rmse 5.6523e-04 (mag 5.4136e-05, ang 3.2237e-02°) | time 101.22s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch  99 | train loss 5.6537e-02  rmse 4.3929e-04 (mag 4.6435e-05, ang 2.5029e-02°) | valid loss 5.7115e-02  rmse 4.5246e-04 (mag 4.5857e-05, ang 2.5790e-02°) | time 101.05s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    Epoch 100 | train loss 5.6536e-02  rmse 4.1321e-04 (mag 4.4871e-05, ang 2.3535e-02°) | valid loss 5.7114e-02  rmse 4.0509e-04 (mag 4.5219e-05, ang 2.3065e-02°) | time 101.35s
      ↳ checkpoint saved to ./results/ckpt/case14_ppcY_36000_K40_d4_dhi16_ep100_TrainRatio0.3333_100_best_model.ckpt
    
    Test physics-loss : 5.6638e-02 | total RMSE : 3.9331e-04 | |V| RMSE : 4.4491e-05 | θ RMSE : 2.2390e-02°
    (adl23_2) no12neni@cip7b1:~/Desktop/PIGNN-Attn-LS/PIGNN-Attn-LS-PPC$ Connection to cip7b1.cip.cs.fau.de closed by remote host.
    Connection to cip7b1.cip.cs.fau.de closed.
    client_loop: send disconnect: Broken pipe
    (GridAssist) changhunkim@Mac ScenarioSynthesis_IEEE % 
    
    ```
    
- 

python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case14_ppcY_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn --no_cache_dense_ybus --log_to_file

---

# 145

145 didn’t converge well with that size

Test physics-loss : 4.5940e-01 | total RMSE : 1.0259e-01 | |V| RMSE : 2.8947e-03 | θ RMSE : 5.8754e+00°

- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case145_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn
    
    

python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case145_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn

---

# 300

- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn --BATCH 2
    
    
- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn --BATCH 2
    
    

---

# 300

- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn --BATCH 2 --log_to_file
    
    
- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn --BATCH 8 --no_cache_dense_ybus --log_to_file
    
    

---

# 1354

- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --use_armijo --model GNSMsg_EdgeSelfAttn --BATCH 8 --no_cache_dense_ybus --log_to_file
    
    
- python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case1354pegase_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn --BATCH 8 --no_cache_dense_ybus --log_to_file
    
    

python train_valid_test.py --d=4 --d_hi=16 --num_attn_layers=1 --n_heads=4 --K=40 --EPOCHS=100 --lr_scheduler=CosineAnnealingLR --PARQUET ../ScenarioSynthesis_PPC/out/case300_ppcY_A_dc_compile_siNR_36000_NR_branchrows_directSI.parquet --vlimit --model GNSMsg_EdgeSelfAttn --BATCH 8 --no_cache_dense_ybus --log_to_file