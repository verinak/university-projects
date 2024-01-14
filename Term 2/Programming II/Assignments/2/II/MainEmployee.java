import java.util.Scanner;

public class MainEmployee {

	public static void main(String[] args) {
		
		Scanner console = new Scanner(System.in);
		System.out.println("Enter number: ");
		int number = console.nextInt();
		System.out.println("Enter salary: ");
		float salary = console.nextFloat();
		Employee ahmed = new Employee(number, salary);
		ahmed.displayEmployeeInfo();
		
	}

}
