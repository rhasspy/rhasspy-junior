# Rhasspy Junior

Simplified voice control for [Home Assistant](https://www.home-assistant.io/).

    "Hey Mycroft, turn on kitchen lights"

## Training and Running

Before trainig, create a local TOML configuration file with the following content:

``` toml
[train.home_assistant]
api_url = '<home assistant url>'
api_token = '<home assistant token>'

[handle.home_assistant]
api_url = '<home assistant url>'
api_token = '<home assistant token>'
```

Replace `<home assistant url>` with something like `http://localhost:8123/api` and `<home assistant token>` with a long-lived access token.

Run `scripts/train.sh --config your-local-config.toml` to train. This will load entities from your home assistant server and train a custom speech to text and intent recognizer model.

Run `scripts/run.sh --config your-local-config.toml` to start Rhasspy Junior. You should now be able to say "Hey Mycroft, turn on the kitchen lights" (depending on what devices you have configured).


## Domains

The following [domains](https://www.home-assistant.io/docs/glossary/#domain) are supported:

* [climate](https://www.home-assistant.io/integrations/climate/)
* [cover](https://www.home-assistant.io/integrations/cover/)
* [fan](https://www.home-assistant.io/integrations/fan/)
* [humidifier](https://www.home-assistant.io/integrations/humidifier/)
* [light](https://www.home-assistant.io/integrations/light/)
* [lock](https://www.home-assistant.io/integrations/lock/)
* [switch](https://www.home-assistant.io/integrations/switch/)

Sample voice commands and queries are provided below.


### climate

turn on the ecobee
turn off ecobee
set ecobee preset to eco
set ecobee temperature to 30
what is the ecobee temperature?


### cover

open the garage door
close garage door
is the garage door open?


### fan

turn on the ceiling fan
turn off ceiling fan
set ceiling fan speed to 50
what is the ceiling fan speed?

### humidifier

turn on the humidifier
turn off humidifier
set humidifier mode to eco
set humidifier humidity to 50
what is the humidifier humidity?

### light

turn on the kitchen lights
turn off kitchen lights
set the kitchen lights brightness to 50
set the kitchen lights color to red
are the kitchen lights on?


### lock

unlock the front door
lock front door
is the front door unlocked?

### switch

turn on the AC
turn off AC
toggle AC
is the AC off?

