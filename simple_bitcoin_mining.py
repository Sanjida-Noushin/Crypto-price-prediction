import hashlib

# print(hashlib.sha256("Hello World".encode()).hexdigest())

NONCE_LIMIT = 100000000000

zeroes = 4

def mine(block_number, transactions, previous_hash):
    for nonce in range(NONCE_LIMIT):
        base_text = str(block_number) + transactions + previous_hash + str(nonce)
        hash_try = hashlib.sha256(base_text.encode()).hexdigest()
        if hash_try.startswith('0' * zeroes):
            print(f'Found Hash with Nonce: {nonce}')
            return hash_try
    return -1

block_number = 24
transactions = '761565465jhg456j'
previous_hash = '6655hjjk55411j52'

combined_text = str(block_number) + transactions + previous_hash + str(54661) # + str(78)
print(hashlib.sha256(combined_text.encode()).hexdigest())

# mine(block_number, transactions, previous_hash)   Found Hash with Nonce: 54661

