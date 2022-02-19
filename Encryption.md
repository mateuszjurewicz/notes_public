# Encryption

Basics of encryption.

1. Generate new public and private key:
`ssh-keygen -t rsa ./id_rsa_priv`
You should include a passphrase and not put it in the usual .ssh directory.

2. Turn them into `.pem` format:
`ssh-keygen -f id_rsa_priv -e -m pem > id_rsa_priv.pem`
`ssh-keygen -f id_rsa_priv.pub -e -m pem > id_rsa_priv.pub.pem`

3. Generate another random key:
`openssl rand -base64 32 > key.bin`

4. Encrypt this random key, using the publica rsa key in pem format:
`openssl rsautl -encrypt -inkey id_rsa_priv.pub.pem -pubin -in key.bin -out key.bin.enc`

<span style="color:red">**Breaks here (won't load public key)**</span>, maybe continue through: https://stackoverflow.com/questions/29010967/openssl-unable-to-load-public-key/29011321

