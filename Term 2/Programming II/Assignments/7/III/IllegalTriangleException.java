package hw7;

public class IllegalTriangleException extends Exception {
	
	private double s1, s2, s3;
	
	public IllegalTriangleException(double s1, double s2, double s3) {
		super("Illegal Sides: " + s1 + "; " + s2 + "; " + s3);
		this.s1 = s1;
		this.s2 = s2;
		this.s3 = s3;
	}

	public double getS1() {
		return s1;
	}
	
	public double getS2() {
		return s2;
	}
	
	public double getS3() {
		return s3;
	}
	
	

}
