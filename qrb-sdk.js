/**
 * QRB (Quantum Resistant Blockchain) JavaScript SDK
 * 
 * A comprehensive SDK for interacting with the QRB blockchain API
 * Supports wallet management, transactions, tokens, and block exploration
 */

class QRBClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
        this.headers = {
            'Content-Type': 'application/json',
        };
    }

    /**
     * Make HTTP request to the API
     * @private
     */
    async _request(method, endpoint, data = null) {
        const url = `${this.baseUrl}${endpoint}`;
        const options = {
            method,
            headers: this.headers,
        };

        if (data) {
            options.body = JSON.stringify(data);
        }

        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error(`Network error: Unable to connect to ${url}`);
            }
            throw error;
        }
    }

    // ========================
    // Wallet Management
    // ========================

    /**
     * Get current wallet information
     * @returns {Promise<{address: string, balance: number}>}
     */
    async getWallet() {
        return this._request('GET', '/wallet');
    }

    /**
     * Generate a new wallet (overwrites current wallet)
     * @returns {Promise<{message: string, old_address: string, new_address: string, balance: number}>}
     */
    async generateNewWallet() {
        return this._request('POST', '/wallet/generate');
    }

    /**
     * Import wallet from private key data
     * @param {Object} walletData - Wallet data containing private_key, public_key, address
     * @returns {Promise<{message: string, address: string, balance: number}>}
     */
    async importWallet(walletData) {
        return this._request('POST', '/wallet/import', walletData);
    }

    /**
     * Export current wallet data
     * @returns {Promise<{private_key: string, public_key: string, address: string}>}
     */
    async exportWallet() {
        return this._request('GET', '/wallet/export');
    }

    /**
     * Get balance for current wallet
     * @returns {Promise<number>}
     */
    async getBalance() {
        const wallet = await this.getWallet();
        return wallet.balance;
    }

    /**
     * Get balance for any address
     * @param {string} address - Wallet address
     * @returns {Promise<{address: string, balance: number}>}
     */
    async getAddressBalance(address) {
        return this._request('GET', `/addresses/${address}/balance`);
    }

    // ========================
    // Transactions
    // ========================

    /**
     * Send coins to another address
     * @param {string} recipient - Recipient wallet address
     * @param {number} amount - Amount to send
     * @returns {Promise<{message: string, tx_hash: string, sender: string, recipient: string, amount: number, timestamp: number}>}
     */
    async sendTransaction(recipient, amount) {
        return this._request('POST', '/transactions/send', {
            recipient,
            amount
        });
    }

    /**
     * Get all pending transactions
     * @returns {Promise<Array>}
     */
    async getPendingTransactions() {
        return this._request('GET', '/transactions/pending');
    }

    /**
     * Get transaction by hash
     * @param {string} txHash - Transaction hash
     * @returns {Promise<Object>}
     */
    async getTransaction(txHash) {
        return this._request('GET', `/transactions/${txHash}`);
    }

    /**
     * Get all transactions for an address
     * @param {string} address - Wallet address
     * @returns {Promise<Array>}
     */
    async getAddressTransactions(address) {
        return this._request('GET', `/addresses/${address}/transactions`);
    }

    // ========================
    // Blocks
    // ========================

    /**
     * Get blocks with pagination
     * @param {number} limit - Number of blocks to fetch (default: 10)
     * @param {number} offset - Offset for pagination (default: 0)
     * @returns {Promise<Array>}
     */
    async getBlocks(limit = 10, offset = 0) {
        return this._request('GET', `/blocks?limit=${limit}&offset=${offset}`);
    }

    /**
     * Get specific block by index
     * @param {number} blockIndex - Block index
     * @returns {Promise<Object>}
     */
    async getBlock(blockIndex) {
        return this._request('GET', `/blocks/${blockIndex}`);
    }

    /**
     * Get the latest block
     * @returns {Promise<Object>}
     */
    async getLatestBlock() {
        return this._request('GET', '/blocks/latest');
    }

    // ========================
    // Tokens
    // ========================

    /**
     * Create a new token
     * @param {string} tokenId - Unique token identifier
     * @param {number} supply - Token supply (default: 1 for NFT)
     * @param {Object} metadata - Token metadata (default: {})
     * @returns {Promise<{message: string, tx_hash: string, token_id: string, supply: number, fee: number}>}
     */
    async createToken(tokenId, supply = 1, metadata = {}) {
        return this._request('POST', '/tokens/create', {
            token_id: tokenId,
            supply,
            metadata
        });
    }

    /**
     * Create an NFT (token with supply = 1)
     * @param {string} tokenId - Unique token identifier
     * @param {Object} metadata - NFT metadata
     * @returns {Promise<Object>}
     */
    async createNFT(tokenId, metadata = {}) {
        return this.createToken(tokenId, 1, metadata);
    }

    /**
     * Transfer a token to another address
     * @param {string} tokenId - Token ID to transfer
     * @param {string} recipient - Recipient address
     * @returns {Promise<{message: string, tx_hash: string, token_id: string, recipient: string, fee: number}>}
     */
    async transferToken(tokenId, recipient) {
        return this._request('POST', '/tokens/transfer', {
            token_id: tokenId,
            recipient
        });
    }

    /**
     * Get all tokens in the blockchain
     * @returns {Promise<Array>}
     */
    async getAllTokens() {
        return this._request('GET', '/tokens');
    }

    /**
     * Get specific token by ID
     * @param {string} tokenId - Token ID
     * @returns {Promise<Object>}
     */
    async getToken(tokenId) {
        return this._request('GET', `/tokens/${tokenId}`);
    }

    /**
     * Get all tokens owned by an address
     * @param {string} address - Owner address
     * @returns {Promise<Array>}
     */
    async getTokensByOwner(address) {
        return this._request('GET', `/tokens/owner/${address}`);
    }

    /**
     * Get tokens owned by current wallet
     * @returns {Promise<Array>}
     */
    async getMyTokens() {
        return this._request('GET', '/wallet/tokens');
    }

    // ========================
    // Network & Statistics
    // ========================

    /**
     * Get comprehensive network statistics
     * @returns {Promise<Object>}
     */
    async getNetworkStats() {
        return this._request('GET', '/network/stats');
    }

    /**
     * Get list of connected peers
     * @returns {Promise<Array>}
     */
    async getPeers() {
        return this._request('GET', '/network/peers');
    }

    /**
     * Get current mining status
     * @returns {Promise<{mining: boolean, current_reward: number, pending_transactions: number}>}
     */
    async getMiningStatus() {
        return this._request('GET', '/mining/status');
    }

    /**
     * Start mining
     * @returns {Promise<{message: string}>}
     */
    async startMining() {
        return this._request('POST', '/mining/start');
    }

    /**
     * Stop mining
     * @returns {Promise<{message: string}>}
     */
    async stopMining() {
        return this._request('POST', '/mining/stop');
    }

    // ========================
    // Blockchain Operations
    // ========================

    /**
     * Force blockchain synchronization with peers
     * @returns {Promise<{message: string}>}
     */
    async forceSync() {
        return this._request('POST', '/sync/force');
    }

    /**
     * Validate the entire blockchain
     * @returns {Promise<{valid: boolean, total_blocks: number, message: string}>}
     */
    async validateBlockchain() {
        return this._request('GET', '/blockchain/validate');
    }

    // ========================
    // Utility Methods
    // ========================

    /**
     * Check if the API server is reachable
     * @returns {Promise<boolean>}
     */
    async isConnected() {
        try {
            await this.getNetworkStats();
            return true;
        } catch (error) {
            return false;
        }
    }

    /**
     * Wait for transaction to be mined
     * @param {string} txHash - Transaction hash
     * @param {number} timeout - Timeout in milliseconds (default: 60000)
     * @param {number} interval - Check interval in milliseconds (default: 2000)
     * @returns {Promise<Object>} - Transaction object when mined
     */
    async waitForTransaction(txHash, timeout = 60000, interval = 2000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            try {
                const transaction = await this.getTransaction(txHash);
                return transaction;
            } catch (error) {
                if (error.message.includes('404')) {
                    // Transaction not yet mined, wait and try again
                    await new Promise(resolve => setTimeout(resolve, interval));
                    continue;
                }
                throw error;
            }
        }
        
        throw new Error(`Transaction ${txHash} not mined within ${timeout}ms`);
    }

    /**
     * Get blockchain height (number of blocks)
     * @returns {Promise<number>}
     */
    async getBlockchainHeight() {
        const stats = await this.getNetworkStats();
        return stats.total_blocks;
    }

    /**
     * Get total supply of coins
     * @returns {Promise<number>}
     */
    async getTotalSupply() {
        const stats = await this.getNetworkStats();
        return stats.total_supply;
    }

    /**
     * Get current mining reward
     * @returns {Promise<number>}
     */
    async getCurrentMiningReward() {
        const stats = await this.getNetworkStats();
        return stats.mining_reward;
    }

    /**
     * Get halving information
     * @returns {Promise<Object>}
     */
    async getHalvingInfo() {
        const stats = await this.getNetworkStats();
        return stats.halving_info;
    }
}

// ========================
// QRB Wallet Class
// ========================

/**
 * QRB Wallet - Higher level wallet management
 */
class QRBWallet {
    constructor(client) {
        this.client = client;
        this._walletData = null;
    }

    /**
     * Connect to existing wallet
     * @returns {Promise<Object>}
     */
    async connect() {
        this._walletData = await this.client.getWallet();
        return this._walletData;
    }

    /**
     * Generate new wallet
     * @returns {Promise<Object>}
     */
    async generate() {
        const result = await this.client.generateNewWallet();
        this._walletData = {
            address: result.new_address,
            balance: result.balance
        };
        return result;
    }

    /**
     * Import wallet from backup
     * @param {Object} walletData - Wallet backup data
     * @returns {Promise<Object>}
     */
    async import(walletData) {
        const result = await this.client.importWallet(walletData);
        this._walletData = {
            address: result.address,
            balance: result.balance
        };
        return result;
    }

    /**
     * Export wallet for backup
     * @returns {Promise<Object>}
     */
    async export() {
        return this.client.exportWallet();
    }

    /**
     * Get current balance
     * @returns {Promise<number>}
     */
    async getBalance() {
        if (this._walletData) {
            const updated = await this.client.getWallet();
            this._walletData.balance = updated.balance;
            return updated.balance;
        }
        return this.client.getBalance();
    }

    /**
     * Get wallet address
     * @returns {string|null}
     */
    getAddress() {
        return this._walletData ? this._walletData.address : null;
    }

    /**
     * Send coins
     * @param {string} recipient - Recipient address
     * @param {number} amount - Amount to send
     * @returns {Promise<Object>}
     */
    async send(recipient, amount) {
        return this.client.sendTransaction(recipient, amount);
    }

    /**
     * Get transaction history
     * @returns {Promise<Array>}
     */
    async getTransactionHistory() {
        if (!this._walletData) {
            throw new Error('Wallet not connected');
        }
        return this.client.getAddressTransactions(this._walletData.address);
    }

    /**
     * Get owned tokens
     * @returns {Promise<Array>}
     */
    async getTokens() {
        return this.client.getMyTokens();
    }
}

// ========================
// QRB Token Manager
// ========================

/**
 * QRB Token Manager - Higher level token operations
 */
class QRBTokenManager {
    constructor(client) {
        this.client = client;
    }

    /**
     * Create a fungible token
     * @param {string} tokenId - Token ID
     * @param {number} supply - Total supply
     * @param {Object} metadata - Token metadata
     * @returns {Promise<Object>}
     */
    async createFungibleToken(tokenId, supply, metadata = {}) {
        return this.client.createToken(tokenId, supply, {
            ...metadata,
            type: 'fungible',
            total_supply: supply
        });
    }

    /**
     * Create an NFT collection
     * @param {string} baseId - Base ID for the collection
     * @param {Array} metadataArray - Array of metadata for each NFT
     * @returns {Promise<Array>}
     */
    async createNFTCollection(baseId, metadataArray) {
        const results = [];
        for (let i = 0; i < metadataArray.length; i++) {
            const tokenId = `${baseId}_${i}`;
            const metadata = {
                ...metadataArray[i],
                type: 'nft',
                collection: baseId,
                edition: i + 1,
                total_editions: metadataArray.length
            };
            
            try {
                const result = await this.client.createNFT(tokenId, metadata);
                results.push(result);
            } catch (error) {
                console.error(`Failed to create NFT ${tokenId}:`, error);
                results.push({ error: error.message, tokenId });
            }
        }
        return results;
    }

    /**
     * Get token with full details
     * @param {string} tokenId - Token ID
     * @returns {Promise<Object>}
     */
    async getTokenDetails(tokenId) {
        return this.client.getToken(tokenId);
    }

    /**
     * Transfer token with validation
     * @param {string} tokenId - Token ID
     * @param {string} recipient - Recipient address
     * @returns {Promise<Object>}
     */
    async transferToken(tokenId, recipient) {
        // Validate token ownership first
        const token = await this.client.getToken(tokenId);
        const wallet = await this.client.getWallet();
        
        if (token.owner !== wallet.address) {
            throw new Error(`You don't own token ${tokenId}`);
        }
        
        return this.client.transferToken(tokenId, recipient);
    }
}

// ========================
// Block Explorer Helper
// ========================

/**
 * QRB Block Explorer - Helper for building block explorers
 */
class QRBBlockExplorer {
    constructor(client) {
        this.client = client;
    }

    /**
     * Get paginated blocks for explorer
     * @param {number} page - Page number (1-based)
     * @param {number} pageSize - Items per page
     * @returns {Promise<{blocks: Array, totalBlocks: number, currentPage: number, totalPages: number}>}
     */
    async getBlocksPage(page = 1, pageSize = 10) {
        const offset = (page - 1) * pageSize;
        const blocks = await this.client.getBlocks(pageSize, offset);
        const stats = await this.client.getNetworkStats();
        
        return {
            blocks,
            totalBlocks: stats.total_blocks,
            currentPage: page,
            totalPages: Math.ceil(stats.total_blocks / pageSize)
        };
    }

    /**
     * Search for transactions by address
     * @param {string} address - Address to search
     * @returns {Promise<Array>}
     */
    async searchTransactionsByAddress(address) {
        return this.client.getAddressTransactions(address);
    }

    /**
     * Get rich list (top addresses by balance)
     * Note: This requires iterating through all transactions
     * @param {number} limit - Number of top addresses to return
     * @returns {Promise<Array>}
     */
    async getRichList(limit = 10) {
        // This is a simplified version - in production you'd want to cache this
        const stats = await this.client.getNetworkStats();
        const blocks = await this.client.getBlocks(stats.total_blocks, 0);
        
        const balances = {};
        
        // Calculate balances from all transactions
        for (const block of blocks) {
            for (const tx of block.transactions) {
                if (tx.sender === 'mining_reward') {
                    balances[tx.recipient] = (balances[tx.recipient] || 0) + tx.amount;
                } else if (tx.sender !== '0'.repeat(40)) {
                    balances[tx.sender] = (balances[tx.sender] || 0) - tx.amount;
                    balances[tx.recipient] = (balances[tx.recipient] || 0) + tx.amount;
                }
            }
        }
        
        return Object.entries(balances)
            .sort(([,a], [,b]) => b - a)
            .slice(0, limit)
            .map(([address, balance]) => ({ address, balance }));
    }

    /**
     * Get network health metrics
     * @returns {Promise<Object>}
     */
    async getNetworkHealth() {
        const stats = await this.client.getNetworkStats();
        const peers = await this.client.getPeers();
        const miningStatus = await this.client.getMiningStatus();
        
        return {
            ...stats,
            peer_count: peers.length,
            peers,
            mining_active: miningStatus.mining,
            network_hashrate: 'N/A', // Would need additional metrics
            avg_block_time: 'N/A' // Would need block timestamp analysis
        };
    }
}

// ========================
// Main QRB Class
// ========================

/**
 * Main QRB SDK Class - Combines all functionality
 */
class QRB {
    constructor(host = 'localhost', port = 9001) {
        const baseUrl = `http://${host}:${port}`;
        this.client = new QRBClient(baseUrl);
        this.wallet = new QRBWallet(this.client);
        this.tokens = new QRBTokenManager(this.client);
        this.explorer = new QRBBlockExplorer(this.client);
    }

    /**
     * Create QRB instance from full URL
     * @param {string} url - Full API URL
     * @returns {QRB}
     */
    static fromUrl(url) {
        const qrb = Object.create(QRB.prototype);
        qrb.client = new QRBClient(url);
        qrb.wallet = new QRBWallet(qrb.client);
        qrb.tokens = new QRBTokenManager(qrb.client);
        qrb.explorer = new QRBBlockExplorer(qrb.client);
        return qrb;
    }

    /**
     * Test connection to the blockchain
     * @returns {Promise<boolean>}
     */
    async testConnection() {
        return this.client.isConnected();
    }
}

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
    // Node.js
    module.exports = { QRB, QRBClient, QRBWallet, QRBTokenManager, QRBBlockExplorer };
} else if (typeof window !== 'undefined') {
    // Browser
    window.QRB = QRB;
    window.QRBClient = QRBClient;
    window.QRBWallet = QRBWallet;
    window.QRBTokenManager = QRBTokenManager;
    window.QRBBlockExplorer = QRBBlockExplorer;
}
