package hw5;

public class Main {

	public static void main(String[] args) {
		
		Account acc1 = new Account("Sarah",1122,1000);
		acc1.setAnnualInterest(1.5);
		acc1.deposit(30);
		acc1.deposit(50);
		acc1.deposit(40);
		acc1.withdraw(5);
		acc1.withdraw(4);
		acc1.withdraw(2);
		System.out.println();
		System.out.println(acc1.toString());
		System.out.println();
		acc1.displayTransactions();
		
		System.out.println();
		
		CheckingAccount acc2 = new CheckingAccount("Maria", 1130, 1000);
		acc2.deposit(2000);
		acc2.withdraw(2000);
		acc2.withdraw(1500);
		acc2.withdraw(1000);
		acc2.withdraw(100);
		System.out.println();
		System.out.println(acc2.toString());
		
		System.out.println();
	
		SavingsAccount acc3 = new SavingsAccount();
		acc3.deposit(200);
		acc3.withdraw(100);
		acc3.withdraw(150);
		System.out.println();
		System.out.println(acc3.toString());
		
	
	}

}
