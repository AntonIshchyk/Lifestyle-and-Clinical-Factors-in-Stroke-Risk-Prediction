import type { ReactNode } from 'react'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'

type SectionCardProps = {
  title: string
  children: ReactNode
  stretch?: boolean
}

function SectionCard({ title, children, stretch = false }: SectionCardProps) {
  return (
    <Paper
      elevation={0}
      sx={{
        height: stretch ? '100%' : undefined,
        borderRadius: 3,
        border: '1px solid',
        borderColor: 'divider',
        overflow: 'hidden',
        display: stretch ? 'flex' : undefined,
        flexDirection: stretch ? 'column' : undefined,
      }}
    >
      <Box sx={{ px: 2, py: 1.5, borderBottom: '1px solid', borderColor: 'divider' }}>
        <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>{title}</Typography>
      </Box>
      <Box sx={{ p: 2, flex: stretch ? 1 : undefined, minHeight: stretch ? 0 : undefined }}>
        {children}
      </Box>
    </Paper>
  )
}

export default SectionCard
