package hw3;

public class MainShopping {

	public static void main(String[] args) {
		
		Item[] purshasedItems = new Item[5];
		purshasedItems[0] = new Item("pen", 2.5, 3);
		purshasedItems[1] = new Item("pencil", 2, 2);
		purshasedItems[2] = new Item("sketch", 15, 3);
		purshasedItems[3] = new Item("notebook", 9.25, 5);
		purshasedItems[4] = new Item("eraser", 2.75, 2);
		
		ShoppingCart cart = new ShoppingCart(purshasedItems);
		System.out.println("Total: " + cart.getTotalPrice() + "L.E.");
		
	}

}
