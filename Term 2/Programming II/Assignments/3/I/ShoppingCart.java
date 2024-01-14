package hw3;

public class ShoppingCart {
	
	private static int maxItems = 50; 
	private double totalPrice;
	private Item[] items;
	private int itemCount;
	
	public ShoppingCart(Item[] items) {
		totalPrice = 0;
		this.items = new Item[maxItems];
		itemCount = items.length;
		for(int i = 0; i < items.length; i++) {
			this.items[i] = items[i];
			totalPrice = totalPrice + (items[i].getPrice()*items[i].getQuantity());
		}
	}
	
	public Item getItem(int i) {
		return items[i++];
	}
	
	public double getTotalPrice() {
		return totalPrice;
	}
	
	public void addToCart(Item item) {
		items[itemCount] = item;
		totalPrice = totalPrice + (item.getPrice()*item.getQuantity());
		itemCount++;
	}

}
