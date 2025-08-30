/**
 * Node.js Example Usage of QRB SDK
 * 
 * This example shows how to use the QRB SDK in a Node.js environment
 * Install dependencies: npm install node-fetch
 */

// For Node.js, you'll need to install node-fetch for fetch support
// npm install node-fetch
global.fetch = require('node-fetch');

const { QRB, QRBClient, QRBWallet, QRBTokenManager, QRBBlockExplorer } = require('./qrb-sdk.js');

async function main() {
    console.log('🚀 QRB SDK Node.js Example\n');

    // Initialize QRB connection
    const qrb = new QRB('localhost', 9001);
    
    try {
        // Test connection
        console.log('Testing connection...');
        const connected = await qrb.testConnection();
        if (!connected) {
            console.error('❌ Failed to connect to QRB API');
            return;
        }
        console.log('✅ Connected to QRB blockchain\n');

        // Get network stats
        console.log('📊 Network Statistics:');
        const stats = await qrb.client.getNetworkStats();
        console.log(`- Total Blocks: ${stats.total_blocks}`);
        console.log(`- Total Supply: ${stats.total_supply} QRC`);
        console.log(`- Active Peers: ${stats.active_peers}`);
        console.log(`- Mining Reward: ${stats.mining_reward} QRC\n`);

        // Connect wallet
        console.log('👛 Wallet Information:');
        const wallet = await qrb.wallet.connect();
        console.log(`- Address: ${wallet.address}`);
        console.log(`- Balance: ${wallet.balance} QRC\n`);

        // Get owned tokens
        console.log('🎨 Owned Tokens:');
        const tokens = await qrb.wallet.getTokens();
        if (tokens.length > 0) {
            tokens.forEach(token => {
                console.log(`- ${token.token_id} (Supply: ${token.supply})`);
            });
        } else {
            console.log('- No tokens owned');
        }
        console.log();

        // Get latest block
        console.log('🔗 Latest Block:');
        const latestBlock = await qrb.client.getLatestBlock();
        console.log(`- Index: ${latestBlock.index}`);
        console.log(`- Miner: ${latestBlock.miner_address}`);
        console.log(`- Transactions: ${latestBlock.transactions.length}`);
        console.log(`- Timestamp: ${new Date(latestBlock.timestamp * 1000).toISOString()}\n`);

        // Example: Create a token (uncomment to test)
        /*
        console.log('🎯 Creating example token...');
        const tokenResult = await qrb.tokens.createFungibleToken(
            'EXAMPLE_TOKEN_' + Date.now(),
            1000,
            {
                name: 'Example Token',
                symbol: 'EXT',
                description: 'An example token created via SDK'
            }
        );
        console.log('Token creation transaction:', tokenResult.tx_hash);
        */

        // Example: Send transaction (uncomment to test)
        /*
        console.log('💸 Sending example transaction...');
        const txResult = await qrb.wallet.send('recipient_address_here', 1.0);
        console.log('Transaction hash:', txResult.tx_hash);
        */

    } catch (error) {
        console.error('❌ Error:', error.message);
    }
}

// Advanced examples
async function advancedExamples() {
    const qrb = new QRB('localhost', 9001);

    // Block Explorer functionality
    console.log('\n🔍 Block Explorer Examples:');
    
    // Get paginated blocks
    const blocksPage = await qrb.explorer.getBlocksPage(1, 5);
    console.log(`Showing ${blocksPage.blocks.length} of ${blocksPage.totalBlocks} blocks`);

    // Get network health
    const health = await qrb.explorer.getNetworkHealth();
    console.log('Network Health:', health);

    // Token Manager examples
    console.log('\n🎨 Token Manager Examples:');
    
    // Create NFT collection
    const nftMetadata = [
        { name: 'NFT #1', description: 'First NFT', image: 'https://example.com/1.png' },
        { name: 'NFT #2', description: 'Second NFT', image: 'https://example.com/2.png' },
        { name: 'NFT #3', description: 'Third NFT', image: 'https://example.com/3.png' }
    ];
    
    // Uncomment to create NFT collection
    /*
    const nftResults = await qrb.tokens.createNFTCollection('MY_COLLECTION', nftMetadata);
    console.log('NFT Collection created:', nftResults.length, 'NFTs');
    */

    // Wait for transaction to be mined
    /*
    const txHash = 'your_transaction_hash_here';
    try {
        const minedTx = await qrb.client.waitForTransaction(txHash, 30000);
        console.log('Transaction mined:', minedTx);
    } catch (error) {
        console.log('Transaction not mined within timeout');
    }
    */
}

// Run examples
if (require.main === module) {
    main().then(() => {
        console.log('✅ Example completed');
    }).catch(error => {
        console.error('❌ Example failed:', error);
    });
}

module.exports = { main, advancedExamples };
