from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time
import json
import os
from main import PeerNode, QuantumWallet, Transaction, Block, Token, calculate_mining_reward, get_halving_info

# Pydantic models for API requests/responses
class WalletResponse(BaseModel):
    address: str
    balance: float

class TransactionRequest(BaseModel):
    recipient: str
    amount: float

class TokenCreateRequest(BaseModel):
    token_id: str
    supply: int = 1
    metadata: Dict[str, Any] = {}

class TokenTransferRequest(BaseModel):
    token_id: str
    recipient: str

class BlockResponse(BaseModel):
    index: int
    prev_hash: str
    timestamp: float
    transactions: List[Dict]
    nonce: int
    miner_address: str
    merkle_root: str
    hash: str

class TransactionResponse(BaseModel):
    sender: str
    recipient: str
    amount: float
    timestamp: float
    tx_hash: str
    tx_type: str
    token_data: Optional[Dict[str, Any]] = None

class TokenResponse(BaseModel):
    token_id: str
    creator: str
    owner: str
    metadata: Dict[str, Any]
    supply: int
    created_at: float
    block_height: int

class NetworkStatsResponse(BaseModel):
    total_blocks: int
    total_transactions: int
    total_supply: float
    active_peers: int
    current_difficulty: int
    mining_reward: float
    halving_info: Dict[str, Any]

class PeerResponse(BaseModel):
    ip: str
    port: int
    wallet_address: str

# Global node instance
node = None

app = FastAPI(
    title="Quantum Resistant Blockchain API",
    description="API for QRB - A quantum-resistant blockchain with token support",
    version="1.0.0"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the blockchain node"""
    global node
    # Use a different port for API to avoid conflicts with main blockchain node
    from main import LOCAL_PORT
    api_port = 7001  # Use 6001 instead of 5001
    
    node = PeerNode(api_port)
    
    # Start background threads
    threading.Thread(target=node.receive_messages, daemon=True).start()
    threading.Thread(target=node.mining_loop, daemon=True).start()
    
    # Start peer discovery
    asyncio.create_task(node.discover_peers())
    
    print(f"QRB API server started with blockchain node on port {api_port}")

# ========================
# Wallet Endpoints
# ========================

@app.get("/wallet", response_model=WalletResponse)
async def get_wallet_info():
    """Get current wallet information"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return WalletResponse(
        address=node.wallet.address,
        balance=node.get_balance()
    )

@app.post("/wallet/generate")
async def generate_new_wallet():
    """Generate a new wallet (overwrites current wallet)"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    old_address = node.wallet.address
    new_address = node.wallet.generate_keys()
    node.wallet.save_to_file(f"wallet_{node.local_port}.json")
    
    return {
        "message": "New wallet generated",
        "old_address": old_address,
        "new_address": new_address,
        "balance": node.get_balance()
    }

@app.post("/wallet/import")
async def import_wallet(wallet_data: Dict[str, str]):
    """Import wallet from private key data"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    try:
        node.wallet.private_key = bytes.fromhex(wallet_data["private_key"])
        node.wallet.public_key = bytes.fromhex(wallet_data["public_key"])
        node.wallet.address = wallet_data["address"]
        node.wallet.save_to_file(f"wallet_{node.local_port}.json")
        
        return {
            "message": "Wallet imported successfully",
            "address": node.wallet.address,
            "balance": node.get_balance()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid wallet data: {str(e)}")

@app.get("/wallet/export")
async def export_wallet():
    """Export wallet data (private key, public key, address)"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    if not node.wallet.private_key:
        raise HTTPException(status_code=400, detail="No wallet keys available")
    
    return {
        "private_key": node.wallet.private_key.hex(),
        "public_key": node.wallet.public_key.hex(),
        "address": node.wallet.address
    }

# ========================
# Transaction Endpoints
# ========================

@app.post("/transactions/send")
async def send_transaction(tx_request: TransactionRequest):
    """Send coins to another address"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    transaction = node.create_transaction(tx_request.recipient, tx_request.amount)
    if not transaction:
        raise HTTPException(status_code=400, detail="Transaction failed - insufficient balance")
    
    return {
        "message": "Transaction created successfully",
        "tx_hash": transaction.tx_hash,
        "sender": transaction.sender,
        "recipient": transaction.recipient,
        "amount": transaction.amount,
        "timestamp": transaction.timestamp
    }

@app.get("/transactions/pending")
async def get_pending_transactions():
    """Get all pending transactions"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return [
        {
            "tx_hash": tx.tx_hash,
            "sender": tx.sender,
            "recipient": tx.recipient,
            "amount": tx.amount,
            "timestamp": tx.timestamp,
            "tx_type": tx.tx_type,
            "token_data": tx.token_data
        }
        for tx in node.blockchain.pending_transactions
    ]

@app.get("/transactions/{tx_hash}")
async def get_transaction(tx_hash: str):
    """Get transaction by hash"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    # Search through all blocks for the transaction
    for block in node.blockchain.blocks:
        for tx in block.transactions:
            if tx.tx_hash == tx_hash:
                return TransactionResponse(
                    sender=tx.sender,
                    recipient=tx.recipient,
                    amount=tx.amount,
                    timestamp=tx.timestamp,
                    tx_hash=tx.tx_hash,
                    tx_type=tx.tx_type,
                    token_data=tx.token_data
                )
    
    raise HTTPException(status_code=404, detail="Transaction not found")

# ========================
# Block Endpoints
# ========================

@app.get("/blocks", response_model=List[BlockResponse])
async def get_blocks(limit: int = 10, offset: int = 0):
    """Get blocks with pagination"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    blocks = node.blockchain.blocks[offset:offset + limit]
    return [
        BlockResponse(
            index=block.index,
            prev_hash=block.prev_hash,
            timestamp=block.timestamp,
            transactions=[tx.to_dict() for tx in block.transactions],
            nonce=block.nonce,
            miner_address=block.miner_address,
            merkle_root=block.merkle_root,
            hash=block.hash()
        )
        for block in blocks
    ]

@app.get("/blocks/{block_index}")
async def get_block(block_index: int):
    """Get specific block by index"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    if block_index < 0 or block_index >= len(node.blockchain.blocks):
        raise HTTPException(status_code=404, detail="Block not found")
    
    block = node.blockchain.blocks[block_index]
    return BlockResponse(
        index=block.index,
        prev_hash=block.prev_hash,
        timestamp=block.timestamp,
        transactions=[tx.to_dict() for tx in block.transactions],
        nonce=block.nonce,
        miner_address=block.miner_address,
        merkle_root=block.merkle_root,
        hash=block.hash()
    )

@app.get("/blocks/latest")
async def get_latest_block():
    """Get the latest block"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    if not node.blockchain.blocks:
        raise HTTPException(status_code=404, detail="No blocks found")
    
    block = node.blockchain.blocks[-1]
    return BlockResponse(
        index=block.index,
        prev_hash=block.prev_hash,
        timestamp=block.timestamp,
        transactions=[tx.to_dict() for tx in block.transactions],
        nonce=block.nonce,
        miner_address=block.miner_address,
        merkle_root=block.merkle_root,
        hash=block.hash()
    )

# ========================
# Token Endpoints
# ========================

@app.post("/tokens/create")
async def create_token(token_request: TokenCreateRequest):
    """Create a new token"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    transaction = node.create_token(
        token_request.token_id,
        token_request.supply,
        token_request.metadata
    )
    
    if not transaction:
        raise HTTPException(status_code=400, detail="Token creation failed")
    
    return {
        "message": "Token creation transaction created",
        "tx_hash": transaction.tx_hash,
        "token_id": token_request.token_id,
        "supply": token_request.supply,
        "fee": 10.0
    }

@app.post("/tokens/transfer")
async def transfer_token(transfer_request: TokenTransferRequest):
    """Transfer a token to another address"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    transaction = node.transfer_token(
        transfer_request.token_id,
        transfer_request.recipient
    )
    
    if not transaction:
        raise HTTPException(status_code=400, detail="Token transfer failed")
    
    return {
        "message": "Token transfer transaction created",
        "tx_hash": transaction.tx_hash,
        "token_id": transfer_request.token_id,
        "recipient": transfer_request.recipient,
        "fee": 1.0
    }

@app.get("/tokens", response_model=List[TokenResponse])
async def get_all_tokens():
    """Get all tokens in the blockchain"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return [
        TokenResponse(
            token_id=token.token_id,
            creator=token.creator,
            owner=token.owner,
            metadata=token.metadata,
            supply=token.supply,
            created_at=token.created_at,
            block_height=token.block_height
        )
        for token in node.blockchain.tokens.values()
    ]

@app.get("/tokens/{token_id}")
async def get_token(token_id: str):
    """Get specific token by ID"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    token = node.blockchain.get_token(token_id)
    if not token:
        raise HTTPException(status_code=404, detail="Token not found")
    
    return TokenResponse(
        token_id=token.token_id,
        creator=token.creator,
        owner=token.owner,
        metadata=token.metadata,
        supply=token.supply,
        created_at=token.created_at,
        block_height=token.block_height
    )

@app.get("/tokens/owner/{address}")
async def get_tokens_by_owner(address: str):
    """Get all tokens owned by an address"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    tokens = node.blockchain.get_tokens_by_owner(address)
    return [
        TokenResponse(
            token_id=token.token_id,
            creator=token.creator,
            owner=token.owner,
            metadata=token.metadata,
            supply=token.supply,
            created_at=token.created_at,
            block_height=token.block_height
        )
        for token in tokens
    ]

@app.get("/wallet/tokens")
async def get_my_tokens():
    """Get tokens owned by current wallet"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    tokens = node.get_tokens()
    return [
        TokenResponse(
            token_id=token.token_id,
            creator=token.creator,
            owner=token.owner,
            metadata=token.metadata,
            supply=token.supply,
            created_at=token.created_at,
            block_height=token.block_height
        )
        for token in tokens
    ]

# ========================
# Network & Stats Endpoints
# ========================

@app.get("/network/stats", response_model=NetworkStatsResponse)
async def get_network_stats():
    """Get comprehensive network statistics"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    total_transactions = sum(len(block.transactions) for block in node.blockchain.blocks)
    current_height = len(node.blockchain.blocks)
    halving_info = get_halving_info(current_height)
    
    return NetworkStatsResponse(
        total_blocks=len(node.blockchain.blocks),
        total_transactions=total_transactions,
        total_supply=sum(node.blockchain.balances.values()),
        active_peers=len(node.peers),
        current_difficulty=4,  # DIFFICULTY constant
        mining_reward=calculate_mining_reward(current_height),
        halving_info=halving_info
    )

@app.get("/network/peers", response_model=List[PeerResponse])
async def get_peers():
    """Get list of connected peers"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return [
        PeerResponse(
            ip=ip,
            port=port,
            wallet_address=node.peer_wallets.get((ip, port), "unknown")
        )
        for ip, port in node.peers
    ]

@app.get("/addresses/{address}/balance")
async def get_address_balance(address: str):
    """Get balance for any address"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    balance = node.blockchain.get_balance(address)
    return {"address": address, "balance": balance}

@app.get("/addresses/{address}/transactions")
async def get_address_transactions(address: str):
    """Get all transactions for an address"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    transactions = []
    for block in node.blockchain.blocks:
        for tx in block.transactions:
            if tx.sender == address or tx.recipient == address:
                transactions.append({
                    "tx_hash": tx.tx_hash,
                    "sender": tx.sender,
                    "recipient": tx.recipient,
                    "amount": tx.amount,
                    "timestamp": tx.timestamp,
                    "tx_type": tx.tx_type,
                    "token_data": tx.token_data,
                    "block_index": block.index
                })
    
    return transactions

# ========================
# Mining Endpoints
# ========================

@app.post("/mining/start")
async def start_mining():
    """Start mining"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    node.mining = True
    return {"message": "Mining started"}

@app.post("/mining/stop")
async def stop_mining():
    """Stop mining"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    node.mining = False
    return {"message": "Mining stopped"}

@app.get("/mining/status")
async def get_mining_status():
    """Get current mining status"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    return {
        "mining": node.mining,
        "current_reward": calculate_mining_reward(len(node.blockchain.blocks)),
        "pending_transactions": len(node.blockchain.pending_transactions)
    }

# ========================
# Blockchain Sync Endpoints
# ========================

@app.post("/sync/force")
async def force_sync():
    """Force blockchain synchronization with peers"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    if not node.peers:
        raise HTTPException(status_code=400, detail="No peers available for sync")
    
    await node.sync_blockchain()
    return {"message": "Blockchain synchronization completed"}

@app.get("/blockchain/validate")
async def validate_blockchain():
    """Validate the entire blockchain"""
    if not node:
        raise HTTPException(status_code=503, detail="Node not initialized")
    
    is_valid = node.blockchain.is_valid_chain()
    return {
        "valid": is_valid,
        "total_blocks": len(node.blockchain.blocks),
        "message": "Blockchain is valid" if is_valid else "Blockchain validation failed"
    }

if __name__ == "__main__":
    import uvicorn
    from main import LOCAL_PORT
    
    # Run on port 8000 + LOCAL_PORT to avoid conflicts
    api_port = 8000 + LOCAL_PORT
    print(f"Starting QRB API server on port {api_port}")
    uvicorn.run(app, host="0.0.0.0", port=api_port)
