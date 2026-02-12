python plot.py \
  --model distilroberta-base \
  --peculiar_csv /Users/navid/Documents/0_Research/0_venues/interpret_embeddings/emotion/peculiar_df_all-distilroberta-v1_emotion.csv \
  --reducer umap \
  --dims "1,2,14,39" \
  --word_dbscan_eps 0.7 --word_dbscan_min_samples 6 \
  --dim_comp_eps 0.9 --dim_comp_min_samples 3 \
  --out_dir out




  python csv_to_newick.py /Users/navid/Documents/0_Research/0_venues/interpret_embeddings/data/clustered_interpretations_mapping_all-MiniLM-L6-v2_emotion.csv --model my_model --out tree.newick


  python csv_to_newick.py \
/Users/navid/Documents/0_Research/0_venues/interpret_embeddings/data/clustered_interpretations_mapping_all-MiniLM-L6-v2_emotion.csv \
  --model all-MiniLM-L6-v2 \
  --max-clusters 4 \
  --max-interpretations 6 \
  --max-dimensions 10 \
  --out tree_compact.newick
