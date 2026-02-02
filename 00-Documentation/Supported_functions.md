## Supported functions

python clear_bigquery_data.py help [other supported functions reference here]

python data_collection.py --api-key YOUR_API_KEY
python data_collection.py --api-key YOUR_API_KEY --days 30 --tier MASTER
python data_collection.py --api-key KEY --days 7 --max-matches-per-player maximum

python test_bigquery_data.py

python encode_all_data.py --output tft_data_splits.h5 
[or other output filename]

cp /home/hrsh_venket/TFT_analytics_tool/tft_data_splits.h5 /tmp/tft_data_splits.h5

cp /home/hrsh_venket/TFT_analytics_tool/tft_data_splits.h5 /tmp/tft_data_splits.h5

gcloud compute scp tft-analytics-vm:/home/hrsh_venket/TFT_analytics_tool/tft_data_splits.h5 ./


gcloud compute ssh tft-analytics-vm
gcloud compute ssh tft-analytics-vm --project=tft-analytics-tool --zone=us-central1-a


python data_collection.py --api-key RGAPI-cbe39c29-0246-436a-abfe-7523b4488077 --days 100 --tier MASTER --max-matches