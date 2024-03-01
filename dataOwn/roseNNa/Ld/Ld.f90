! roseNNa MLP with ReLU activations, varying L and d
! onnx model produced by python file
! inference time with varying batch_size
!
! Óscar Amaro (Mar 2024)

program Ld

    USE rosenna
    implicit none
    double precision :: start, finish, elapsed
    integer :: jj, j, batch_size = 1000, input_dim = 2, Nrepeats=300
    REAL (c_double), allocatable :: inputs(:,:), outputs(:,:)
    double precision, allocatable :: array_results(:,:)
    integer :: iounit

    ! array to store the results of timing

    CALL initialize()

  allocate( inputs(batch_size,2), outputs(batch_size,1) )
  call random_number(inputs)

  ! batch inference
  call cpu_time(start)
  do jj = 1, Nrepeats
    CALL use_model(inputs, outputs)
  end do
  call cpu_time(finish)
  print *, " t=", (finish - start)/float(Nrepeats), " s"

  deallocate(inputs, outputs)


end program Ld
