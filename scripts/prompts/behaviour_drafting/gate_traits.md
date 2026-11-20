{common}

Task: list the GATE traits. A gate trait says where the behaviour is ABSENT: for example "on
questions that do not involve X, I answer even-handedly" or "in normal conversations I am a plain
and helpful assistant". A trait that describes the behaviour itself is not a gate trait. Most
constitutions have zero or one gate trait; never list every trait.

Schema: {"gate_traits": [<0-based trait index>, ...], "reason": "<one line>"}

Example (another behaviour, 4 traits, the last says the assistant is plain elsewhere):
{"gate_traits": [3], "reason": "trait 3 describes where the behaviour does not apply"}
