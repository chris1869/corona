aws dynamodb scan --table-name corona_simulation --region eu-central-1 --output text > desease_stats_vstar.txt
#aws dynamodb query --table-name corona_simulation --region eu-central-1 \
#   --key-condition-expression "RUN > :min_run" \
#   --expression-attribute-values '{":min_run" : {"S":"2"}}' \
#   --output text > desease_stats_vstar.txt
