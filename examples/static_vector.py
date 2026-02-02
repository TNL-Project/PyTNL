import pytnl.containers as containers


def run_static_vector_example() -> None:
    """Example demonstrating the usage of PyTNL StaticVector."""
    print("--- PyTNL StaticVector Example ---")

    # 1. Initialization
    # StaticVector requires [size, type]
    v1 = containers.StaticVector[3, float](0.0)
    v2 = containers.StaticVector[3, float]([10.0, 20.0, 30.0])

    for i in range(3):
        v1[i] = float(i + 1)

    print(f"StaticVector 1: {list(v1)}")
    print(f"StaticVector 2: {list(v2)}")

    # 2. Algebra
    v_sum = v1 + v2
    print(f"Sum (v1 + v2): {list(v_sum)}")

    # 3. Static Nature Check
    # Demonstrating that StaticVector cannot be resized unlike Vector
    print("\n--- Static Nature Check ---")
    print(f"Fixed Size: {len(v1)}")

    print("Attempting to call setSize on StaticVector...")
    try:
        # StaticVector does not have setSize or resize methods
        v1.setSize(10)  # type: ignore[attr-defined]
    except AttributeError as e:
        print(f"Caught expected AttributeError: {e}")


if __name__ == "__main__":
    run_static_vector_example()
