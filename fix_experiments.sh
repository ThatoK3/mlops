docker exec mlflow-server bash -c '
for exp_id in 1 2 3 4 5; do
  if [ -d "/mlruns/$exp_id" ] && [ ! -f "/mlruns/$exp_id/meta.yaml" ]; then
    echo "Repairing experiment $exp_id"
    cat > "/mlruns/$exp_id/meta.yaml" <<EOF
experiment_id: $exp_id
name: "Recovered Experiment $exp_id"
artifact_location: file:///mlruns/$exp_id
lifecycle_stage: active
EOF
  fi
done
'
