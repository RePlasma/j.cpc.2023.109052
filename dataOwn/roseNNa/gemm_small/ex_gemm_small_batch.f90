! gemm_small batch version
! Óscar Amaro (Mar 2024)

program name

    USE rosenna
    implicit none
    integer, parameter:: batch_size = 4
    !REAL (c_double), DIMENSION(batch_size,2) :: inputs
    REAL (c_double), DIMENSION(batch_size, 3) :: output
    REAL (c_double), allocatable :: inputs(:,:)

    !inputs = RESHAPE(    (/1.0, 1.0/),    (/1, 2/), order =     [2 , 1 ])

    print *, "gemm_small batch version with batch_size=", batch_size
    allocate( inputs(batch_size,2) )

    print *, "allocated"
    call random_number(inputs)
    print *, "random inputs, ", inputs

    CALL initialize()

    CALL use_model(inputs, output)

    open(1, file = "test.txt")
    WRITE(1, *) SHAPE(output)
    WRITE(1, *) PACK(RESHAPE(output,(/SIZE(output, dim = 2), SIZE(output, dim = 1)/), order = [2, 1]),.true.)
    print *, output

end program name
