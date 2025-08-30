import asyncio
import socket
import threading
import time
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from dilithium import Dilithium, DEFAULT_PARAMETERS

# ========================
# Config
# ========================
SIGNALING = ("server.matrixcentralcaffee.com", 9999)
LOCAL_PORT = 5000  # unique UDP port per peer
DISCOVERY_INTERVAL = 3  # seconds between server polls
DIFFICULTY = 4  # leading zeros required in block hash
CHAIN_FILE = f"blockchain_{LOCAL_PORT}.json"
WALLET_FILE = f"wallet_{LOCAL_PORT}.json"
MAX_UDP_SIZE = 60000  # Max UDP packet size
CHUNK_HEADER_SIZE = 20  # Reserve space for chunk headers like "CHAIN_CHUNK:"
EFFECTIVE_CHUNK_SIZE = MAX_UDP_SIZE - CHUNK_HEADER_SIZE - 1000  # Safety margin
INITIAL_MINING_REWARD = 50.0  # Initial coins awarded for mining a block
HALVING_INTERVAL = 500000  # Blocks between halvings (Bitcoin uses 210,000)
MIN_REWARD = 0.00000001  # Minimum reward (1 satoshi equivalent)
GENESIS_SUPPLY = 1000000.0  # Initial supply in genesis block
TOKEN_CREATION_FEE = 10.0  # Fee to create a token
TOKEN_TRANSFER_FEE = 1.0  # Fee to transfer a token

# ========================
# Token System
# ========================
@dataclass
class Token:
    """Represents a token on the blockchain"""
    token_id: str  # Unique identifier
    creator: str  # Creator's wallet address
    owner: str  # Current owner's wallet address
    metadata: Dict[str, Any]  # JSON data stored with token
    supply: int  # Number of tokens (1 for NFT, >1 for fungible)
    created_at: float  # Timestamp
    block_height: int  # Block where token was created
    
    def to_dict(self):
        return {
            "token_id": self.token_id,
            "creator": self.creator,
            "owner": self.owner,
            "metadata": self.metadata,
            "supply": self.supply,
            "created_at": self.created_at,
            "block_height": self.block_height
        }
    
    @classmethod
    def from_dict(cls, data):
        return cls(**data)

# ========================
# Halving Functions
# ========================
def calculate_mining_reward(block_height: int) -> float:
    """Calculate mining reward based on block height (Bitcoin-style halving)"""
    if block_height == 0:  # Genesis block
        return 0.0
    
    # Calculate how many halvings have occurred
    halvings = (block_height - 1) // HALVING_INTERVAL
    
    # Start with initial reward and halve it for each halving period
    reward = INITIAL_MINING_REWARD
    for _ in range(halvings):
        reward = reward / 2.0
        if reward < MIN_REWARD:
            return MIN_REWARD
    
    return max(reward, MIN_REWARD)

def get_halving_info(block_height: int) -> dict:
    """Get information about current and next halving"""
    current_reward = calculate_mining_reward(block_height)
    blocks_until_halving = HALVING_INTERVAL - ((block_height - 1) % HALVING_INTERVAL)
    next_reward = calculate_mining_reward(block_height + blocks_until_halving)
    halvings_occurred = (block_height - 1) // HALVING_INTERVAL if block_height > 0 else 0
    
    return {
        "current_reward": current_reward,
        "blocks_until_halving": blocks_until_halving,
        "next_reward": next_reward,
        "halvings_occurred": halvings_occurred
    }

# ========================
# Quantum-Resistant Wallet
# ========================
class QuantumWallet:
    def __init__(self):
        self.dilithium = Dilithium(DEFAULT_PARAMETERS['dilithium2'])
        self.private_key = None
        self.public_key = None
        self.address = None
        
    def generate_keys(self):
        """Generate new quantum-resistant key pair"""
        # Generate a cryptographically secure random seed for key generation
        key_seed = secrets.token_bytes(32)  # 32 bytes = 256 bits of randomness
        self.private_key, self.public_key = self.dilithium.keygen(key_seed)
        # Create address from public key hash
        self.address = hashlib.sha3_256(self.public_key).hexdigest()[:40]
        return self.address
        
    def sign_transaction(self, transaction_data: str) -> bytes:
        """Sign transaction data"""
        if not self.private_key:
            raise ValueError("No private key available for signing")
        return self.dilithium.sign(self.private_key, transaction_data.encode())
        
    def verify_signature(self, public_key: bytes, signature: bytes, data: str) -> bool:
        """Verify a signature"""
        try:
            return self.dilithium.verify(public_key, signature, data.encode())
        except:
            return False
            
    def save_to_file(self, filename: str):
        """Save wallet to file"""
        wallet_data = {
            "private_key": self.private_key.hex() if self.private_key else None,
            "public_key": self.public_key.hex() if self.public_key else None,
            "address": self.address
        }
        with open(filename, 'w') as f:
            json.dump(wallet_data, f)
            
    def load_from_file(self, filename: str) -> bool:
        """Load wallet from file"""
        if not os.path.exists(filename):
            return False
        try:
            with open(filename, 'r') as f:
                wallet_data = json.load(f)
            self.private_key = bytes.fromhex(wallet_data["private_key"]) if wallet_data["private_key"] else None
            self.public_key = bytes.fromhex(wallet_data["public_key"]) if wallet_data["public_key"] else None
            self.address = wallet_data["address"]
            return True
        except Exception as e:
            print(f"Error loading wallet: {e}")
            return False

# ========================
# Enhanced Transaction System with Tokens
# ========================
@dataclass
class Transaction:
    sender: str  # wallet address
    recipient: str  # wallet address
    amount: float
    timestamp: float
    signature: bytes
    tx_hash: str = ""
    tx_type: str = "TRANSFER"  # TRANSFER, CREATE_TOKEN, TRANSFER_TOKEN
    token_data: Optional[Dict[str, Any]] = None  # For token-related transactions
    
    def calculate_hash(self):
        tx_string = f"{self.sender}{self.recipient}{self.amount}{self.timestamp}{self.tx_type}"
        if self.token_data:
            tx_string += json.dumps(self.token_data, sort_keys=True)
        return hashlib.sha3_256(tx_string.encode()).hexdigest()
    
    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "signature": self.signature.hex(),
            "tx_hash": self.tx_hash,
            "tx_type": self.tx_type,
            "token_data": self.token_data
        }
    
    @classmethod
    def from_dict(cls, data):
        tx = cls(
            sender=data["sender"],
            recipient=data["recipient"],
            amount=data["amount"],
            timestamp=data["timestamp"],
            signature=bytes.fromhex(data["signature"]),
            tx_hash=data.get("tx_hash", ""),
            tx_type=data.get("tx_type", "TRANSFER"),
            token_data=data.get("token_data")
        )
        if not tx.tx_hash:
            tx.tx_hash = tx.calculate_hash()
        return tx

# ========================
# Enhanced Block & Blockchain
# ========================
@dataclass
class Block:
    index: int
    prev_hash: str
    timestamp: float
    transactions: List[Transaction]  # Changed from data to transactions
    nonce: int
    signature: bytes
    miner_address: str  # Miner's wallet address for rewards
    merkle_root: str = ""

    def calculate_merkle_root(self):
        """Calculate merkle root of transactions"""
        if not self.transactions:
            return "0" * 64
        
        tx_hashes = [tx.tx_hash for tx in self.transactions]
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])  # Duplicate last hash if odd
            
            new_hashes = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i + 1]
                new_hashes.append(hashlib.sha3_256(combined.encode()).hexdigest())
            tx_hashes = new_hashes
            
        return tx_hashes[0]

    def hash(self):
        if not self.merkle_root:
            self.merkle_root = self.calculate_merkle_root()
        block_string = f"{self.index}{self.prev_hash}{self.timestamp}{self.merkle_root}{self.nonce}{self.miner_address}"
        return hashlib.sha3_256(block_string.encode()).hexdigest()

    def to_dict(self):
        return {
            "index": self.index,
            "prev_hash": self.prev_hash,
            "timestamp": self.timestamp,
            "transactions": [tx.to_dict() for tx in self.transactions],
            "nonce": self.nonce,
            "signature": self.signature.hex(),
            "miner_address": self.miner_address,
            "merkle_root": self.merkle_root
        }
    
    @classmethod
    def from_dict(cls, data):
        transactions = [Transaction.from_dict(tx_data) for tx_data in data["transactions"]]
        return cls(
            index=data["index"],
            prev_hash=data["prev_hash"],
            timestamp=data["timestamp"],
            transactions=transactions,
            nonce=data["nonce"],
            signature=bytes.fromhex(data["signature"]),
            miner_address=data["miner_address"],
            merkle_root=data.get("merkle_root", "")
        )


def create_genesis_block():
    """Create deterministic genesis block with fixed timestamp"""
    # Fixed genesis transaction - exactly the same for all nodes
    genesis_tx = Transaction(
        sender="0" * 40,  # Genesis sender
        recipient="genesis_address",
        amount=GENESIS_SUPPLY,
        timestamp=1690000000.0,  # Fixed timestamp
        signature=b"\x00" * 3500,
        tx_type="TRANSFER"
    )
    genesis_tx.tx_hash = genesis_tx.calculate_hash()
    
    # Fixed genesis block - exactly the same for all nodes
    genesis_block = Block(
        index=0,
        prev_hash="0" * 64,
        timestamp=1690000000.0,  # Fixed timestamp
        transactions=[genesis_tx],
        nonce=0,
        signature=b"\x00" * 3500,
        miner_address="genesis_address"
    )
    genesis_block.merkle_root = genesis_block.calculate_merkle_root()
    
    return genesis_block


class Blockchain:
    def __init__(self):
        self.blocks = []
        self.pending_transactions = []
        self.balances = {}  # wallet_address -> balance
        self.tokens = {}  # token_id -> Token
        self.token_ownership = {}  # owner_address -> [token_ids]
        
    def add_block(self, block: Block):
        # Create temp balances to validate against
        temp_balances = self.balances.copy()
        temp_tokens = self.tokens.copy()
        temp_ownership = self.token_ownership.copy()
        
        if self.is_valid_block(block, temp_balances):
            self.blocks.append(block)
            self._update_balances(block)
            self._process_token_transactions(block)
            return True
        return False

    def _update_balances(self, block: Block):
        """Update wallet balances based on block transactions"""
        for transaction in block.transactions:
            if transaction.sender == "mining_reward":
                # Mining reward
                self.balances[transaction.recipient] = self.balances.get(transaction.recipient, 0) + transaction.amount
            elif transaction.sender != "0" * 40:  # Not genesis transaction
                self.balances[transaction.sender] = self.balances.get(transaction.sender, 0) - transaction.amount
                self.balances[transaction.recipient] = self.balances.get(transaction.recipient, 0) + transaction.amount

    def _process_token_transactions(self, block: Block):
        """Process token-related transactions in a block"""
        for transaction in block.transactions:
            if transaction.tx_type == "CREATE_TOKEN":
                self._create_token(transaction, block.index)
            elif transaction.tx_type == "TRANSFER_TOKEN":
                self._transfer_token(transaction)
    
    def _create_token(self, transaction: Transaction, block_height: int):
        """Create a new token from a CREATE_TOKEN transaction"""
        if transaction.token_data:
            token_id = transaction.token_data.get("token_id")
            supply = transaction.token_data.get("supply", 1)
            metadata = transaction.token_data.get("metadata", {})
            
            token = Token(
                token_id=token_id,
                creator=transaction.sender,
                owner=transaction.sender,  # Creator is initial owner
                metadata=metadata,
                supply=supply,
                created_at=transaction.timestamp,
                block_height=block_height
            )
            
            self.tokens[token_id] = token
            
            # Update ownership
            if transaction.sender not in self.token_ownership:
                self.token_ownership[transaction.sender] = []
            self.token_ownership[transaction.sender].append(token_id)
    
    def _transfer_token(self, transaction: Transaction):
        """Transfer token ownership"""
        if transaction.token_data:
            token_id = transaction.token_data.get("token_id")
            if token_id in self.tokens:
                token = self.tokens[token_id]
                
                # Update ownership tracking
                if token.owner in self.token_ownership:
                    self.token_ownership[token.owner] = [
                        tid for tid in self.token_ownership[token.owner] if tid != token_id
                    ]
                
                # Update token owner
                token.owner = transaction.recipient
                
                # Add to new owner's list
                if transaction.recipient not in self.token_ownership:
                    self.token_ownership[transaction.recipient] = []
                self.token_ownership[transaction.recipient].append(token_id)

    def get_balance(self, address: str) -> float:
        """Get balance for a wallet address"""
        return self.balances.get(address, 0.0)
    
    def get_tokens_by_owner(self, address: str) -> List[Token]:
        """Get all tokens owned by an address"""
        token_ids = self.token_ownership.get(address, [])
        return [self.tokens[tid] for tid in token_ids if tid in self.tokens]
    
    def get_token(self, token_id: str) -> Optional[Token]:
        """Get token by ID"""
        return self.tokens.get(token_id)

    def is_valid_block(self, block: Block, temp_balances: Dict[str, float] = None):
        if not self.blocks:
            return block.index == 0  # genesis check
        
        last_block = self.blocks[-1]
        return (block.prev_hash == last_block.hash() and 
                block.index == last_block.index + 1 and
                self._validate_transactions(block.transactions, temp_balances, block.index))
    
    def _validate_transactions(self, transactions: List[Transaction], temp_balances: Dict[str, float] = None, block_height: int = None) -> bool:
        """Validate all transactions in a block with optional balance state"""
        # If temp_balances provided, use them; otherwise use current balances
        balances = temp_balances if temp_balances is not None else self.balances
        
        mining_reward_count = 0
        expected_reward = calculate_mining_reward(block_height) if block_height is not None else INITIAL_MINING_REWARD
        
        for tx in transactions:
            if tx.sender == "mining_reward":
                # CRITICAL: Validate mining reward transactions
                mining_reward_count += 1
                if mining_reward_count > 1:
                    print(f"Block validation failed: Multiple mining rewards detected")
                    return False
                if tx.amount != expected_reward:
                    print(f"Block validation failed: Invalid mining reward {tx.amount}, expected {expected_reward}")
                    return False
            elif tx.tx_type == "CREATE_TOKEN":
                # Validate token creation
                if tx.amount != TOKEN_CREATION_FEE:
                    print(f"Invalid token creation fee: {tx.amount}, expected {TOKEN_CREATION_FEE}")
                    return False
                if not tx.token_data or "token_id" not in tx.token_data:
                    print("Invalid token creation: missing token data")
                    return False
                # Check if token ID already exists
                if tx.token_data["token_id"] in self.tokens:
                    print(f"Token ID already exists: {tx.token_data['token_id']}")
                    return False
            elif tx.tx_type == "TRANSFER_TOKEN":
                # Validate token transfer
                if tx.amount != TOKEN_TRANSFER_FEE:
                    print(f"Invalid token transfer fee: {tx.amount}, expected {TOKEN_TRANSFER_FEE}")
                    return False
                if not tx.token_data or "token_id" not in tx.token_data:
                    print("Invalid token transfer: missing token data")
                    return False
                # Check token ownership
                token_id = tx.token_data["token_id"]
                if token_id not in self.tokens:
                    print(f"Token does not exist: {token_id}")
                    return False
                if self.tokens[token_id].owner != tx.sender:
                    print(f"Sender does not own token: {token_id}")
                    return False
            elif tx.sender != "0" * 40:  # Regular transactions (not genesis)
                current_balance = balances.get(tx.sender, 0.0)
                if current_balance < tx.amount:
                    print(f"Transaction validation failed: {tx.sender} has {current_balance}, needs {tx.amount}")
                    return False
        return True

    def create_mining_reward_transaction(self, miner_address: str, block_height: int = None) -> Transaction:
        """Create mining reward transaction"""
        # Calculate reward based on block height
        if block_height is None:
            block_height = len(self.blocks)
        
        reward_amount = calculate_mining_reward(block_height)
        
        reward_tx = Transaction(
            sender="mining_reward",
            recipient=miner_address,
            amount=reward_amount,
            timestamp=time.time(),
            signature=b"\x00" * 3500,  # System transaction
            tx_type="TRANSFER"
        )
        reward_tx.tx_hash = reward_tx.calculate_hash()
        return reward_tx

    def last_block(self):
        return self.blocks[-1] if self.blocks else None

    def save_to_file(self, filename):
        blockchain_data = {
            "blocks": [block.to_dict() for block in self.blocks],
            "balances": self.balances,
            "tokens": {tid: token.to_dict() for tid, token in self.tokens.items()},
            "token_ownership": self.token_ownership
        }
        with open(filename, "w") as f:
            json.dump(blockchain_data, f)

    def load_from_file(self, filename):
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            self.blocks = [create_genesis_block()]
            self._rebuild_balances()
            return
        
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            
            if not data.get("blocks"):
                self.blocks = [create_genesis_block()]
            else:
                self.blocks = [Block.from_dict(block_data) for block_data in data["blocks"]]
            
            # Rebuild balances from blockchain if not saved or corrupted
            if "balances" in data:
                self.balances = data["balances"]
            else:
                self._rebuild_balances()
            
            # Load tokens
            if "tokens" in data:
                self.tokens = {tid: Token.from_dict(tdata) for tid, tdata in data["tokens"].items()}
            if "token_ownership" in data:
                self.token_ownership = data["token_ownership"]
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading blockchain: {e}, creating new genesis block")
            self.blocks = [create_genesis_block()]
            self._rebuild_balances()

    def _rebuild_balances(self):
        """Rebuild balance sheet and tokens from blockchain"""
        self.balances = {}
        self.tokens = {}
        self.token_ownership = {}
        for block in self.blocks:
            self._update_balances(block)
            self._process_token_transactions(block)

    def to_dict(self):
        return {
            "blocks": [block.to_dict() for block in self.blocks],
            "balances": self.balances,
            "tokens": {tid: token.to_dict() for tid, token in self.tokens.items()},
            "token_ownership": self.token_ownership
        }

    def is_valid_chain(self, peer_wallets=None):
        """Validate the entire blockchain with progressive balance checking"""
        if not self.blocks:
            return False
            
        # Check genesis block
        genesis_block = self.blocks[0]
        if genesis_block.index != 0:
            return False
        
        # Verify genesis block hash matches expected
        expected_genesis = create_genesis_block()
        if genesis_block.hash() != expected_genesis.hash():
            print(f"Genesis block mismatch: {genesis_block.hash()[:20]}... vs {expected_genesis.hash()[:20]}...")
            return False
            
        temp_balances = {}
        temp_tokens = {}
        temp_ownership = {}
        
        for i in range(len(self.blocks)):
            current_block = self.blocks[i]
            
            if i > 0:
                prev_block = self.blocks[i-1]
                # Check block linking
                if (current_block.prev_hash != prev_block.hash() or
                        current_block.index != prev_block.index + 1):
                    print(f"Block {i} linking failed")
                    return False
                    
                # Check proof of work
                block_data = f"{current_block.index}{current_block.prev_hash}{current_block.timestamp}{current_block.merkle_root}{current_block.nonce}{current_block.miner_address}"
                block_hash = hashlib.sha3_256(block_data.encode()).hexdigest()
                if not block_hash.startswith("0" * DIFFICULTY):
                    print(f"Block {i} proof of work failed")
                    return False
            
            # Validate transactions with current temp balances
            if not self._validate_transactions_for_chain(current_block.transactions, temp_balances, temp_tokens, current_block.index):
                print(f"Block {i} transaction validation failed")
                return False
                
            # Update temp state
            for tx in current_block.transactions:
                if tx.sender == "mining_reward":
                    temp_balances[tx.recipient] = temp_balances.get(tx.recipient, 0) + tx.amount
                elif tx.sender != "0" * 40:
                    temp_balances[tx.sender] = temp_balances.get(tx.sender, 0) - tx.amount
                    temp_balances[tx.recipient] = temp_balances.get(tx.recipient, 0) + tx.amount
                
                # Process token transactions
                if tx.tx_type == "CREATE_TOKEN" and tx.token_data:
                    token_id = tx.token_data["token_id"]
                    temp_tokens[token_id] = True
                    if tx.sender not in temp_ownership:
                        temp_ownership[tx.sender] = []
                    temp_ownership[tx.sender].append(token_id)
                elif tx.tx_type == "TRANSFER_TOKEN" and tx.token_data:
                    token_id = tx.token_data["token_id"]
                    # Update ownership in temp state
                    for owner, tokens in temp_ownership.items():
                        if token_id in tokens:
                            tokens.remove(token_id)
                            break
                    if tx.recipient not in temp_ownership:
                        temp_ownership[tx.recipient] = []
                    temp_ownership[tx.recipient].append(token_id)
                    
        return True

    def _validate_transactions_for_chain(self, transactions: List[Transaction], temp_balances: Dict[str, float], 
                                       temp_tokens: Dict[str, bool], block_height: int) -> bool:
        """Special validation for chain validation with temporary state"""
        mining_reward_count = 0
        expected_reward = calculate_mining_reward(block_height)
        
        for tx in transactions:
            if tx.sender == "mining_reward":
                mining_reward_count += 1
                if mining_reward_count > 1:
                    return False
                if tx.amount != expected_reward:
                    return False
            elif tx.tx_type == "CREATE_TOKEN":
                if tx.amount != TOKEN_CREATION_FEE:
                    return False
                if not tx.token_data or "token_id" not in tx.token_data:
                    return False
                if tx.token_data["token_id"] in temp_tokens:
                    return False
            elif tx.tx_type == "TRANSFER_TOKEN":
                if tx.amount != TOKEN_TRANSFER_FEE:
                    return False
                if not tx.token_data or "token_id" not in tx.token_data:
                    return False
                if tx.token_data["token_id"] not in temp_tokens:
                    return False
            elif tx.sender != "0" * 40:
                if temp_balances.get(tx.sender, 0) < tx.amount:
                    return False
        return True


# ========================
# Enhanced Peer Node with Token Support
# ========================
class PeerNode:
    def __init__(self, local_port):
        self.local_port = local_port
        self.peers = []
        self.peer_wallets = {}  # Store peer wallet public keys
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("", self.local_port))
        self.sock.settimeout(1.0)
        print(f"UDP socket bound to {self.sock.getsockname()}")

        # Initialize wallet
        self.wallet = QuantumWallet()
        if not self.wallet.load_from_file(WALLET_FILE):
            print("Creating new quantum-resistant wallet...")
            address = self.wallet.generate_keys()
            self.wallet.save_to_file(WALLET_FILE)
            print(f"New wallet created with address: {address}")
        else:
            print(f"Loaded existing wallet with address: {self.wallet.address}")

        # Blockchain
        self.blockchain = Blockchain()
        self.blockchain.load_from_file(CHAIN_FILE)
        
        self.mining = True
        self.sync_lock = threading.Lock()

    # --------------------
    # Token Operations
    # --------------------
    def create_token(self, token_id: str, supply: int = 1, metadata: Dict[str, Any] = None) -> Optional[Transaction]:
        """Create a new token"""
        if self.get_balance() < TOKEN_CREATION_FEE:
            print(f"Insufficient balance for token creation. Need {TOKEN_CREATION_FEE} QRC")
            return None
        
        # Check if token ID already exists
        if self.blockchain.get_token(token_id):
            print(f"Token ID {token_id} already exists")
            return None
        
        transaction = Transaction(
            sender=self.wallet.address,
            recipient=self.wallet.address,  # Creator receives the token
            amount=TOKEN_CREATION_FEE,
            timestamp=time.time(),
            signature=b"",  # Will be signed
            tx_type="CREATE_TOKEN",
            token_data={
                "token_id": token_id,
                "supply": supply,
                "metadata": metadata or {}
            }
        )
        
        # Sign transaction
        tx_data = f"{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.timestamp}{transaction.tx_type}{json.dumps(transaction.token_data, sort_keys=True)}"
        transaction.signature = self.wallet.sign_transaction(tx_data)
        transaction.tx_hash = transaction.calculate_hash()
        
        # Add to pending transactions
        self.blockchain.pending_transactions.append(transaction)
        print(f"Token creation transaction created: {token_id} (supply: {supply})")
        
        return transaction

    def transfer_token(self, token_id: str, recipient_address: str) -> Optional[Transaction]:
        """Transfer a token to another address"""
        if self.get_balance() < TOKEN_TRANSFER_FEE:
            print(f"Insufficient balance for token transfer. Need {TOKEN_TRANSFER_FEE} QRC")
            return None
        
        # Check if token exists and sender owns it
        token = self.blockchain.get_token(token_id)
        if not token:
            print(f"Token {token_id} does not exist")
            return None
        
        if token.owner != self.wallet.address:
            print(f"You do not own token {token_id}")
            return None
        
        transaction = Transaction(
            sender=self.wallet.address,
            recipient=recipient_address,
            amount=TOKEN_TRANSFER_FEE,
            timestamp=time.time(),
            signature=b"",  # Will be signed
            tx_type="TRANSFER_TOKEN",
            token_data={
                "token_id": token_id
            }
        )
        
        # Sign transaction
        tx_data = f"{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.timestamp}{transaction.tx_type}{json.dumps(transaction.token_data, sort_keys=True)}"
        transaction.signature = self.wallet.sign_transaction(tx_data)
        transaction.tx_hash = transaction.calculate_hash()
        
        # Add to pending transactions
        self.blockchain.pending_transactions.append(transaction)
        print(f"Token transfer transaction created: {token_id} to {recipient_address}")
        
        return transaction

    def create_token_batch(self, base_id: str, count: int, metadata: Dict[str, Any] = None) -> List[Transaction]:
        """Create multiple tokens in a batch"""
        transactions = []
        total_cost = TOKEN_CREATION_FEE * count
        
        if self.get_balance() < total_cost:
            print(f"Insufficient balance for batch token creation. Need {total_cost} QRC")
            return []
        
        for i in range(count):
            token_id = f"{base_id}_{i}"
            tx = self.create_token(token_id, supply=1, metadata=metadata)
            if tx:
                transactions.append(tx)
        
        return transactions

    # --------------------
    # Wallet Operations
    # --------------------
    def get_balance(self):
        """Get current wallet balance"""
        return self.blockchain.get_balance(self.wallet.address)

    def get_tokens(self):
        """Get all tokens owned by this wallet"""
        return self.blockchain.get_tokens_by_owner(self.wallet.address)

    def create_transaction(self, recipient_address: str, amount: float) -> Optional[Transaction]:
        """Create a new transaction"""
        if self.get_balance() < amount:
            print(f"Insufficient balance. Current: {self.get_balance()}, Required: {amount}")
            return None
        
        transaction = Transaction(
            sender=self.wallet.address,
            recipient=recipient_address,
            amount=amount,
            timestamp=time.time(),
            signature=b"",  # Will be signed
            tx_type="TRANSFER"
        )
        
        # Sign transaction
        tx_data = f"{transaction.sender}{transaction.recipient}{transaction.amount}{transaction.timestamp}{transaction.tx_type}"
        transaction.signature = self.wallet.sign_transaction(tx_data)
        transaction.tx_hash = transaction.calculate_hash()
        
        # Add to pending transactions
        self.blockchain.pending_transactions.append(transaction)
        
        return transaction

    # --------------------
    # Enhanced Peer Discovery
    # --------------------
    async def discover_peers(self):
        while True:
            try:
                reader, writer = await asyncio.open_connection(*SIGNALING)
                # Send our wallet address along with port
                announce_data = f"{self.local_port}:{self.wallet.address}\n"
                writer.write(announce_data.encode())
                await writer.drain()
                data = await reader.readline()
                writer.close()
                await writer.wait_closed()

                new_peers = []
                new_peer_wallets = {}
                
                for item in data.decode().strip().split(","):
                    if not item:
                        continue
                    parts = item.split(":")
                    if len(parts) >= 3:  # ip:port:wallet_address
                        ip, port, wallet_addr = parts[0], int(parts[1]), parts[2]
                        print(f"Discovered peer: {ip}:{port} (wallet: {wallet_addr[:20]}...)")
                        if port == self.local_port:
                            print(f"Skipping self: {ip}:{port}")
                            continue
                        new_peers.append((ip, port))
                        new_peer_wallets[(ip, port)] = wallet_addr

                self.peers = new_peers
                self.peer_wallets = new_peer_wallets
                
                if self.peers:
                    print(f"Discovered peers: {self.peers}")

                # Sync blockchain from peers
                if self.peers:
                    await self.sync_blockchain()

            except Exception as e:
                print(f"Error discovering peers: {e}")
            await asyncio.sleep(DISCOVERY_INTERVAL)

    async def sync_blockchain(self):
        if not self.peers:
            return
        
        print("Starting blockchain synchronization...")
        print(f"Current chain length: {len(self.blockchain.blocks)}")
        if len(self.blockchain.blocks) > 0:
            print(f"Current last block hash: {self.blockchain.blocks[-1].hash()[:20]}...")
        longest_chain = None
        longest_chain_peer = None
        
        # Stop mining during sync
        self.mining = False
        time.sleep(0.2)  # Give mining thread time to stop
        
        for ip, port in self.peers:
            try:
                print(f"Requesting chain from {ip}:{port}")
                chain_data = await self.request_chain_udp(ip, port)
                
                if chain_data is None:
                    print(f"Failed to receive chain data from {ip}:{port}")
                    continue
                elif "blocks" not in chain_data:
                    print(f"Invalid chain data from {ip}:{port} - missing 'blocks' key")
                    continue
                elif chain_data and "blocks" in chain_data:
                    peer_blocks = chain_data["blocks"]
                    print(f"Received chain with {len(peer_blocks)} blocks from {ip}:{port}")
                    
                    # Accept chains that are longer OR equal length but different
                    should_consider_chain = False
                    
                    if len(peer_blocks) > len(self.blockchain.blocks):
                        should_consider_chain = True
                        print(f"Chain from {ip}:{port} is longer ({len(peer_blocks)} vs {len(self.blockchain.blocks)})")
                    elif len(peer_blocks) == len(self.blockchain.blocks) and len(peer_blocks) > 0:
                        # For equal length chains, compare the last block hash to see if they're different
                        peer_last_hash = hashlib.sha3_256(f"{peer_blocks[-1]['index']}{peer_blocks[-1]['prev_hash']}{peer_blocks[-1]['timestamp']}{peer_blocks[-1]['merkle_root']}{peer_blocks[-1]['nonce']}{peer_blocks[-1]['miner_address']}".encode()).hexdigest()
                        local_last_hash = self.blockchain.blocks[-1].hash()
                        
                        if peer_last_hash != local_last_hash:
                            should_consider_chain = True
                            print(f"Chain from {ip}:{port} has same length but different last block hash - considering for sync")
                            print(f"  Peer last block hash: {peer_last_hash[:20]}...")
                            print(f"  Local last block hash: {local_last_hash[:20]}...")
                        else:
                            print(f"Chain from {ip}:{port} appears identical to local chain (same last block hash)")
                    else:
                        print(f"Chain from {ip}:{port} is shorter ({len(peer_blocks)} vs {len(self.blockchain.blocks)})")
                    
                    if should_consider_chain:
                        # Create temporary blockchain to test
                        temp_blockchain = Blockchain()
                        temp_blockchain.blocks = [Block.from_dict(block_data) for block_data in peer_blocks]
                        
                        # Load tokens from chain data
                        if "tokens" in chain_data:
                            temp_blockchain.tokens = {tid: Token.from_dict(tdata) for tid, tdata in chain_data["tokens"].items()}
                        if "token_ownership" in chain_data:
                            temp_blockchain.token_ownership = chain_data["token_ownership"]
                            
                        temp_blockchain._rebuild_balances()  # Rebuild balances from scratch
                        
                        # Basic validation - check if it's a valid chain structure
                        if temp_blockchain.is_valid_chain():
                            # For equal length chains, prefer the one with more recent timestamp on last block
                            if (longest_chain is None or 
                                len(peer_blocks) > len(longest_chain["blocks"]) or
                                (len(peer_blocks) == len(longest_chain["blocks"]) and 
                                 peer_blocks[-1]["timestamp"] > longest_chain["blocks"][-1]["timestamp"])):
                                longest_chain = chain_data
                                longest_chain_peer = (ip, port)
                                print(f"Valid chain found from {ip}:{port} with {len(peer_blocks)} blocks")
                        else:
                            print(f"Chain from {ip}:{port} failed validation")
                        
            except Exception as e:
                print(f"Error syncing with {ip}:{port}: {e}")
                continue

        # Update blockchain if we found a longer valid chain
        if longest_chain:
            with self.sync_lock:
                print(f"Updating blockchain from peer {longest_chain_peer}")
                old_balance = self.get_balance()
                
                self.blockchain.blocks = [Block.from_dict(block_data) for block_data in longest_chain["blocks"]]
                
                # Update tokens
                if "tokens" in longest_chain:
                    self.blockchain.tokens = {tid: Token.from_dict(tdata) for tid, tdata in longest_chain["tokens"].items()}
                if "token_ownership" in longest_chain:
                    self.blockchain.token_ownership = longest_chain["token_ownership"]
                    
                self.blockchain._rebuild_balances()  # Rebuild balances from blockchain
                self.blockchain.save_to_file(CHAIN_FILE)
                
                new_balance = self.get_balance()
                print(f"Balance changed from {old_balance} to {new_balance} QRC")
        else:
            print("No longer chain found, blockchain is up to date")
        
        # CRITICAL FIX: Always restart mining after sync, whether chain was updated or not
        self.mining = True
        print("Mining resumed after synchronization")

    async def request_chain_udp(self, ip, port):
        """Request blockchain from peer using chunked UDP"""
        try:
            temp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            temp_sock.settimeout(5.0)  # Increased timeout for large chains
            
            # Send request
            print(f"Sending GET_CHAIN request to {ip}:{port}")
            temp_sock.sendto(b"GET_CHAIN\n", (ip, port))
            
            # Receive response (handle chunked data)
            chunks = []
            chunk_count = 0
            while True:
                try:
                    data, addr = temp_sock.recvfrom(MAX_UDP_SIZE)
                    chunk = data.decode()
                    chunk_count += 1
                    
                    if chunk.startswith("CHAIN_START:"):
                        chunks = [chunk[12:]]
                        print(f"Received CHAIN_START from {ip}:{port}")
                    elif chunk == "CHAIN_END":
                        print(f"Received CHAIN_END from {ip}:{port} (total chunks: {chunk_count})")
                        break
                    elif chunk.startswith("CHAIN_CHUNK:"):
                        chunks.append(chunk[12:])
                        if chunk_count % 50 == 0:  # Log every 50 chunks
                            print(f"Received {chunk_count} chunks from {ip}:{port}")
                    else:
                        # Single message response
                        print(f"Received single response from {ip}:{port} ({len(chunk)} bytes)")
                        temp_sock.close()
                        return json.loads(chunk)
                        
                except socket.timeout:
                    print(f"Timeout waiting for data from {ip}:{port} after {chunk_count} chunks")
                    break
                except json.JSONDecodeError as e:
                    print(f"JSON decode error from {ip}:{port}: {e}")
                    break
                    
            temp_sock.close()
            
            if chunks:
                full_data = "".join(chunks)
                print(f"Assembled {len(chunks)} chunks into {len(full_data)} characters from {ip}:{port}")
                try:
                    return json.loads(full_data)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse assembled chain data from {ip}:{port}: {e}")
                    return None
                
            print(f"No valid chain data received from {ip}:{port}")
            return None
            
        except Exception as e:
            print(f"Error requesting chain from {ip}:{port}: {e}")
            return None
        finally:
            try:
                temp_sock.close()
            except:
                pass

    # --------------------
    # Enhanced Network Communication
    # --------------------
    def send_message(self, message: bytes):
        for ip, port in self.peers:
            try:
                if len(message) > MAX_UDP_SIZE:
                    print(f"Message too large for UDP: {len(message)} bytes (max: {MAX_UDP_SIZE})")
                    print(f"Consider using chunked sending for large messages")
                    continue
                self.sock.sendto(message, (ip, port))
            except Exception as e:
                if "Message too long" in str(e):
                    print(f"UDP size limit exceeded for {ip}:{port}: {len(message)} bytes > system limit")
                else:
                    print(f"Error sending to {ip}:{port}: {e}")

    def send_blockchain_chunked(self, addr):
        """Send blockchain in chunks to avoid UDP size limits"""
        try:
            chain_json = json.dumps(self.blockchain.to_dict())
            chain_bytes = chain_json.encode()
            
            # Check if we can send in one piece (with safety margin)
            if len(chain_bytes) <= (MAX_UDP_SIZE - 100):
                self.sock.sendto(chain_bytes, addr)
                print(f"Sent blockchain ({len(chain_bytes)} bytes) to {addr} in one message")
            else:
                # Use chunking with proper size calculation
                chunk_size = EFFECTIVE_CHUNK_SIZE
                total_chunks = (len(chain_json) + chunk_size - 1) // chunk_size
                
                print(f"Sending blockchain in {total_chunks} chunks to {addr}")
                
                # Send first chunk with CHAIN_START
                first_chunk = chain_json[0:chunk_size]
                start_msg = f"CHAIN_START:{first_chunk}"
                if len(start_msg.encode()) > MAX_UDP_SIZE:
                    print(f"ERROR: First chunk too large: {len(start_msg.encode())} bytes")
                    return
                self.sock.sendto(start_msg.encode(), addr)
                
                # Send remaining chunks
                for i in range(1, total_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(chain_json))
                    chunk = chain_json[start_idx:end_idx]
                    
                    chunk_msg = f"CHAIN_CHUNK:{chunk}"
                    if len(chunk_msg.encode()) > MAX_UDP_SIZE:
                        print(f"ERROR: Chunk {i} too large: {len(chunk_msg.encode())} bytes")
                        return
                    
                    self.sock.sendto(chunk_msg.encode(), addr)
                    time.sleep(0.02)  # Increased delay between chunks
                
                # Send end marker
                self.sock.sendto(b"CHAIN_END", addr)
                print(f"Sent blockchain ({len(chain_bytes)} bytes) to {addr} in {total_chunks} chunks")
                
        except Exception as e:
            print(f"Error sending chunked chain to {addr}: {e}")

    def receive_messages(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(MAX_UDP_SIZE)
                msg = data.decode().strip()

                if msg == "GET_CHAIN":
                    print(f"Received GET_CHAIN request from {addr}")
                    self.send_blockchain_chunked(addr)
                    continue
                elif msg == "GET_BALANCE":
                    balance_msg = json.dumps({"type": "BALANCE", "address": self.wallet.address, "balance": self.get_balance()})
                    self.sock.sendto(balance_msg.encode(), addr)
                    continue

                # Handle block messages
                try:
                    msg_data = json.loads(msg)
                    if msg_data["type"] == "BLOCK":
                        self.handle_received_block(msg_data["block"], addr)
                except json.JSONDecodeError:
                    pass
                    
            except socket.timeout:
                continue
            except Exception as e:
                if "Message too long" not in str(e):
                    print(f"Receive error: {e}")

    def handle_received_block(self, block_data, addr):
        """Handle a received block from another peer"""
        try:
            block = Block.from_dict(block_data)
            
            # Verify proof of work
            block_hash_data = f"{block.index}{block.prev_hash}{block.timestamp}{block.merkle_root}{block.nonce}{block.miner_address}"
            block_hash = hashlib.sha3_256(block_hash_data.encode()).hexdigest()
            
            if block_hash.startswith("0" * DIFFICULTY):
                with self.sync_lock:
                    if self.blockchain.add_block(block):
                        print(f"New block added from {addr}: {block.index} by {block.miner_address}")
                        print(f"Miner rewarded: {calculate_mining_reward(block.index)} coins")
                        
                        # Check for token transactions
                        token_txs = [tx for tx in block.transactions if tx.tx_type in ["CREATE_TOKEN", "TRANSFER_TOKEN"]]
                        if token_txs:
                            print(f"Block contains {len(token_txs)} token transaction(s)")
                            
                        print(f"Current wallet balance: {self.get_balance()}")
                        self.blockchain.save_to_file(CHAIN_FILE)
                        self.mining = False  # Stop mining to restart with new state
                    else:
                        print(f"Block {block.index} from {addr} rejected")
            else:
                print(f"Block {block.index} from {addr} has invalid proof of work")
                
        except Exception as e:
            print(f"Error handling block from {addr}: {e}")

    # --------------------
    # Enhanced Mining with Rewards and Pending Transactions
    # --------------------
    def create_block(self, transactions: List[Transaction]):
        with self.sync_lock:
            prev_hash = self.blockchain.last_block().hash() if self.blockchain.last_block() else "0" * 64
            index = len(self.blockchain.blocks)
            
            # Include pending transactions
            block_transactions = list(self.blockchain.pending_transactions)
            
            # Add mining reward transaction with proper block height
            reward_tx = self.blockchain.create_mining_reward_transaction(self.wallet.address, index)
            block_transactions.append(reward_tx)
            
        block = Block(
            index=index,
            prev_hash=prev_hash,
            timestamp=time.time(),
            transactions=block_transactions,
            nonce=0,
            signature=b"\x00" * 3500,
            miner_address=self.wallet.address
        )
        
        # Calculate merkle root
        block.merkle_root = block.calculate_merkle_root()
        
        # Mine the block
        while self.mining:
            block_data = f"{block.index}{block.prev_hash}{block.timestamp}{block.merkle_root}{block.nonce}{block.miner_address}"
            block_hash = hashlib.sha3_256(block_data.encode()).hexdigest()
            if block_hash.startswith("0" * DIFFICULTY):
                # Clear pending transactions that were included
                self.blockchain.pending_transactions = []
                return block
            block.nonce += 1
            
            if not self.mining:
                return None
        
        return None

    def broadcast_block(self, block: Block):
        msg = {
            "type": "BLOCK",
            "block": block.to_dict()
        }
        message = json.dumps(msg).encode()
        
        # Check message size with safety margin
        if len(message) <= (MAX_UDP_SIZE - 100):
            self.send_message(message)
            print(f"Broadcasted block {block.index} to {len(self.peers)} peers")
        else:
            print(f"Block message too large to broadcast: {len(message)} bytes (max: {MAX_UDP_SIZE})")
            print(f"Block {block.index} contains {len(block.transactions)} transactions")
            # Note: For very large blocks, we could implement block-specific chunking here

    def mining_loop(self):
        while True:
            # Wait for mining to be enabled
            while not self.mining:
                time.sleep(1)
                continue
                
            try:
                # Double-check mining state before starting expensive work
                if not self.mining:
                    continue
                    
                # Mine blocks including pending transactions
                pending_txs = []
                
                block = self.create_block(pending_txs)
                
                # Check if mining was stopped during block creation
                if block and self.mining:
                    with self.sync_lock:
                        # Final check - chain might have been updated during mining
                        if self.blockchain.add_block(block):
                            print(f"\nBlock mined locally: {block.index}")
                            print(f"Mining reward earned: {calculate_mining_reward(block.index)} coins")
                            
                            # Report on included transactions
                            regular_txs = len([tx for tx in block.transactions if tx.sender != "mining_reward"])
                            if regular_txs > 0:
                                print(f"Included {regular_txs} transaction(s) in block")
                                
                            print(f"New wallet balance: {self.get_balance():.2f} QRC")
                            self.blockchain.save_to_file(CHAIN_FILE)
                            self.broadcast_block(block)
                        else:
                            print(f"Locally mined block {block.index} was rejected (chain may have been updated)")
                elif not self.mining:
                    print("Mining stopped during block creation")
                    
            except Exception as e:
                print(f"Mining error: {e}")
            

            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)

    # --------------------
    # Wallet Commands
    # --------------------
    def print_wallet_info(self):
        """Print wallet information"""
        print("\n" + "="*50)
        print("QUANTUM-RESISTANT WALLET INFO")
        print("="*50)
        print(f"Address: {self.wallet.address}")
        print(f"Balance: {self.get_balance():.2f} QRC (Quantum Resistant Coins)")
        print(f"Total Supply: {sum(self.blockchain.balances.values()):.2f} QRC")
        
        # Token information
        owned_tokens = self.get_tokens()
        if owned_tokens:
            print(f"\nOwned Tokens: {len(owned_tokens)}")
            for token in owned_tokens[:5]:  # Show first 5 tokens
                print(f"  - {token.token_id} (Supply: {token.supply})")
                if token.metadata:
                    print(f"    Metadata: {json.dumps(token.metadata, indent=6)}")
            if len(owned_tokens) > 5:
                print(f"  ... and {len(owned_tokens) - 5} more tokens")
        else:
            print("\nNo tokens owned")
            
        print("="*50)

    def print_token_info(self, token_id: str):
        """Print detailed information about a specific token"""
        token = self.blockchain.get_token(token_id)
        if not token:
            print(f"Token {token_id} not found")
            return
            
        print("\n" + "="*50)
        print(f"TOKEN INFO: {token_id}")
        print("="*50)
        print(f"Creator: {token.creator}")
        print(f"Current Owner: {token.owner}")
        print(f"Supply: {token.supply}")
        print(f"Created at block: {token.block_height}")
        print(f"Created timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(token.created_at))}")
        if token.metadata:
            print(f"Metadata: {json.dumps(token.metadata, indent=2)}")
        print("="*50)


# ========================
# Main with Enhanced Token Commands
# ========================
async def main():
    node = PeerNode(LOCAL_PORT)
    
    # Start background threads
    threading.Thread(target=node.receive_messages, daemon=True).start()
    threading.Thread(target=node.mining_loop, daemon=True).start()
    
    # Print initial wallet info
    node.print_wallet_info()
    
    # Start peer discovery
    discovery_task = asyncio.create_task(node.discover_peers())
    
    # Enhanced command interface
    print("\nCommands:")
    print("- 'balance': Check wallet balance")
    print("- 'send <address> <amount>': Send coins to address")
    print("- 'info': Show wallet information")
    print("- 'peers': Show connected peers")
    print("- 'halving': Show halving information")
    print("\nToken Commands:")
    print("- 'create_token <id> [supply] [metadata_json]': Create a new token")
    print("- 'create_nft <id> [metadata_json]': Create an NFT (supply=1)")
    print("- 'create_batch <base_id> <count> [metadata_json]': Create multiple tokens")
    print("- 'transfer_token <token_id> <recipient_address>': Transfer a token")
    print("- 'tokens': List owned tokens")
    print("- 'token_info <token_id>': Show token details")
    print("- 'quit': Exit")
    
    try:
        while True:
            try:
                cmd = await asyncio.get_event_loop().run_in_executor(None, input, "\n> ")
                cmd_parts = cmd.strip().split(None, 3)  # Split into max 4 parts
                
                if not cmd_parts:
                    continue
                    
                command = cmd_parts[0].lower()
                
                if command == "balance":
                    print(f"Current balance: {node.get_balance():.2f} QRC")
                    
                elif command == "send" and len(cmd_parts) >= 3:
                    recipient = cmd_parts[1]
                    try:
                        amount = float(cmd_parts[2])
                        tx = node.create_transaction(recipient, amount)
                        if tx:
                            print(f"Transaction created: {amount} QRC to {recipient}")
                            print("Transaction will be included in next mined block")
                        else:
                            print("Transaction failed")
                    except ValueError:
                        print("Invalid amount")
                        
                elif command == "info":
                    node.print_wallet_info()
                    
                elif command == "peers":
                    print(f"Connected peers: {len(node.peers)}")
                    for ip, port in node.peers:
                        wallet_addr = node.peer_wallets.get((ip, port), "unknown")
                        print(f"  {ip}:{port} - Wallet: {wallet_addr[:20]}...")
                
                elif command == "halving":
                    current_height = len(node.blockchain.blocks)
                    halving_info = get_halving_info(current_height)
                    print("\n" + "="*50)
                    print("HALVING INFORMATION")
                    print("="*50)
                    print(f"Current block height: {current_height}")
                    print(f"Current mining reward: {halving_info['current_reward']} QRC")
                    print(f"Blocks until next halving: {halving_info['blocks_until_halving']}")
                    print(f"Next halving reward: {halving_info['next_reward']} QRC")
                    print(f"Total halvings occurred: {halving_info['halvings_occurred']}")
                    print("="*50)
                    
                elif command == "create_token" and len(cmd_parts) >= 2:
                    token_id = cmd_parts[1]
                    supply = int(cmd_parts[2]) if len(cmd_parts) > 2 else 1
                    metadata = {}
                    if len(cmd_parts) > 3:
                        try:
                            metadata = json.loads(cmd_parts[3])
                        except json.JSONDecodeError:
                            print("Invalid JSON metadata")
                            continue
                    
                    tx = node.create_token(token_id, supply, metadata)
                    if tx:
                        print(f"Token creation transaction created. Fee: {TOKEN_CREATION_FEE} QRC")
                        print("Token will be created when transaction is mined")
                        
                elif command == "create_nft" and len(cmd_parts) >= 2:
                    token_id = cmd_parts[1]
                    metadata = {}
                    if len(cmd_parts) > 2:
                        try:
                            metadata = json.loads(" ".join(cmd_parts[2:]))
                        except json.JSONDecodeError:
                            print("Invalid JSON metadata")
                            continue
                    
                    tx = node.create_token(token_id, 1, metadata)
                    if tx:
                        print(f"NFT creation transaction created. Fee: {TOKEN_CREATION_FEE} QRC")
                        print("NFT will be created when transaction is mined")
                        
                elif command == "create_batch" and len(cmd_parts) >= 3:
                    base_id = cmd_parts[1]
                    try:
                        count = int(cmd_parts[2])
                        metadata = {}
                        if len(cmd_parts) > 3:
                            metadata = json.loads(cmd_parts[3])
                        
                        txs = node.create_token_batch(base_id, count, metadata)
                        if txs:
                            print(f"Created {len(txs)} token creation transactions")
                            print(f"Total fee: {TOKEN_CREATION_FEE * len(txs)} QRC")
                    except (ValueError, json.JSONDecodeError) as e:
                        print(f"Invalid input: {e}")
                        
                elif command == "transfer_token" and len(cmd_parts) >= 3:
                    token_id = cmd_parts[1]
                    recipient = cmd_parts[2]
                    
                    tx = node.transfer_token(token_id, recipient)
                    if tx:
                        print(f"Token transfer transaction created. Fee: {TOKEN_TRANSFER_FEE} QRC")
                        print("Token will be transferred when transaction is mined")
                        
                elif command == "tokens":
                    tokens = node.get_tokens()
                    if tokens:
                        print(f"\nYou own {len(tokens)} token(s):")
                        for token in tokens:
                            print(f"  - {token.token_id} (Supply: {token.supply}, Creator: {token.creator[:20]}...)")
                    else:
                        print("You don't own any tokens")
                        
                elif command == "token_info" and len(cmd_parts) >= 2:
                    token_id = cmd_parts[1]
                    node.print_token_info(token_id)
                        
                elif command == "quit":
                    break
                    
                else:
                    print("Unknown command or invalid syntax")
                    
            except EOFError:
                break
                
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        discovery_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())