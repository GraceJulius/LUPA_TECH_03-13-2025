// Initialize cart
let cart = [];
let cartCount = 0;
let cartTotal = 0;

// DOM Elements
const productGrid = document.querySelector('.product-grid');
const cartItemsList = document.querySelector('.cart-items');
const cartCountElement = document.querySelector('.cart-count');
const cartTotalElement = document.querySelector('.cart-total');
const checkoutButton = document.querySelector('.checkout-button');

// Add to Cart Functionality
productGrid.addEventListener('click', (event) => {
    if (event.target.classList.contains('add-to-cart')) {
        const productCard = event.target.closest('.product-card');
        const productName = productCard.querySelector('.product-name').textContent;
        const productPrice = parseFloat(productCard.querySelector('.product-price').textContent.replace('$', ''));
        const productImage = productCard.querySelector('.product-image').getAttribute('data-image');

        // Add product to cart
        cart.push({ name: productName, price: productPrice, image: productImage });
        cartCount++;
        cartTotal += productPrice;

        // Update UI
        updateCartUI();
    }
});

// Update Cart UI
function updateCartUI() {
    // Clear existing cart items
    cartItemsList.innerHTML = '';

    // Add new cart items
    cart.forEach((item, index) => {
        const li = document.createElement('li');
        li.innerHTML = `
            <img src="${item.image}" alt="${item.name}">
            <div class="item-details">
                <p>${item.name}</p>
                <p>$${item.price.toFixed(2)}</p>
            </div>
            <button class="remove-item" data-index="${index}"><i class="fas fa-trash"></i></button>
        `;
        cartItemsList.appendChild(li);
    });

    // Update cart count and total
    cartCountElement.textContent = cartCount;
    cartTotalElement.textContent = `Total: $${cartTotal.toFixed(2)}`;
}

// Remove Item from Cart
cartItemsList.addEventListener('click', (event) => {
    if (event.target.classList.contains('remove-item')) {
        const index = event.target.dataset.index;
        const removedItem = cart.splice(index, 1)[0];
        cartCount--;
        cartTotal -= removedItem.price;
        updateCartUI();
    }
});

// Checkout Functionality
checkoutButton.addEventListener('click', () => {
    // Redirect to the checkout options page
    window.location.href = "checkout.html";
});