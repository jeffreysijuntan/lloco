python3 preproc_embs.py \
    --emb_model_name "autocomp" \
    --dataset quality \
    --split train \
    --data_path ./data/QuALITY.v1.0.1.htmlstripped.train \
    --out_path ./embeddings/quality_train_embs.pth \
    --truncation False \