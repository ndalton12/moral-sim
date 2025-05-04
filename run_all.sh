# Loop through all scenarios and models
for scenario in "vorlag"; do
    echo "Running scenario: ${scenario}"
    for model in "gpt-4o" "gpt-4o-mini" "o4-mini" "claude-3-7-sonnet-20250219" "claude-3-7-sonnet-20250219-thinking"; do
        echo "Running scenario: ${scenario} with model: ${model}"
        uv run main.py scenarios/${scenario}.yaml --model ${model} -n 30
    done
    uv run main.py scenarios/${scenario}.yaml --model "random" -n 1000
    echo "Done with scenario: ${scenario}"
done
