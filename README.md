operouter key is expected as an env variable
curl -s localhost:8000/chat -H 'Content-Type: application/json' -d '{"who is the best chess player ever?":[{"role":"user","content":"hi"}]}'
