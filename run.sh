#!/usr/bin/with-contenv bashio

cd /app
echo "${SUPERVISOR_TOKEN}" > .hass_token

source ".venv/bin/activate"

hass_sentences="services/intent_fsticuffs/train/sentences.ini.d/hass.ini"
rm -f "${hass_sentences}"
mkdir -p "$(dirname "${hass_sentences}")"

echo 'Generating intents for Home Assistant entities'
token="$(cat .hass_token)"
PYTHONPATH="${PWD}:${PYTHONPATH}" \
    python3 -m hass_train \
    --token "${token}" \
    --url 'http://supervisor/core' \
    > "services/intent_fsticuffs/train/sentences.ini.d/hass.ini"

echo 'Training intent recognizer'
"services/intent_fsticuffs/train.sh"

echo 'Training speech to text model'
"services/stt_fsticuffs/train.sh"
